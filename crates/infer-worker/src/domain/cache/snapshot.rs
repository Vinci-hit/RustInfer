//! Owned recurrent checkpoints. KV suffix retention is a separate concern.
use super::*;

/// Copies only the participating request slots, including convolution history
/// and FP32 delta-rule state. Source and checkpoint never share storage.
/// Scoped copies run in stream order; legacy capture/restore are synchronous.
pub struct LinearSnapshot<T: Dtype, D: LlmBackend> {
    slots: Vec<usize>,
    layers: Vec<LinearLayerState<T, D>>,
}

impl<T: Dtype, D: LlmBackend> LinearSnapshot<T, D> {
    /// Reserve backup storage at startup. Only participating slots are copied.
    pub fn reserve(layers: &[LinearLayerState<T, D>], capacity: usize) -> OpResult<Self> {
        Ok(Self {
            slots: Vec::with_capacity(capacity),
            layers: layers
                .iter()
                .map(|l| LinearLayerState::new(l.dims, capacity, l.conv.device()))
                .collect::<OpResult<_>>()?,
        })
    }

    pub fn capture_on(
        &mut self,
        layers: &[LinearLayerState<T, D>],
        slots: &[usize],
        scope: &D::Scope,
    ) -> OpResult<()> {
        if slots.is_empty()
            || layers.len() != self.layers.len()
            || slots
                .iter()
                .enumerate()
                .any(|(i, s)| slots[..i].contains(s))
            || layers.iter().zip(&self.layers).any(|(src, dst)| {
                src.dims != dst.dims
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(dst.conv.device())
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(
                            infer_core::exec::ExecScope::device(scope),
                        )
                    || slots.len() > dst.conv.shape()[0]
                    || slots.iter().any(|&s| s >= src.conv.shape()[0])
            })
        {
            return Err(OpError::Shape("invalid recurrent snapshot geometry".into()));
        }
        self.slots.clear();
        self.slots.extend_from_slice(slots);
        for (src, dst) in layers.iter().zip(&self.layers) {
            for (row, &slot) in slots.iter().enumerate() {
                D::copy_tensor(
                    scope,
                    &src.conv.narrow(0, slot, 1)?,
                    &mut dst.conv.narrow(0, row, 1)?,
                )?;
                D::copy_tensor(
                    scope,
                    &src.ssm.narrow(0, slot, 1)?,
                    &mut dst.ssm.narrow(0, row, 1)?,
                )?;
            }
        }
        Ok(())
    }

    pub fn restore_on(
        &self,
        layers: &mut [LinearLayerState<T, D>],
        scope: &D::Scope,
    ) -> OpResult<()> {
        if layers.len() != self.layers.len()
            || layers.iter().zip(&self.layers).any(|(dst, src)| {
                dst.dims != src.dims
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(dst.conv.device())
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(
                            infer_core::exec::ExecScope::device(scope),
                        )
                    || self.slots.iter().any(|&s| s >= dst.conv.shape()[0])
            })
        {
            return Err(OpError::Shape("recurrent snapshot layout mismatch".into()));
        }
        for (dst, src) in layers.iter_mut().zip(&self.layers) {
            for (row, &slot) in self.slots.iter().enumerate() {
                D::copy_tensor(
                    scope,
                    &src.conv.narrow(0, row, 1)?,
                    &mut dst.conv.narrow(0, slot, 1)?,
                )?;
                D::copy_tensor(
                    scope,
                    &src.ssm.narrow(0, row, 1)?,
                    &mut dst.ssm.narrow(0, slot, 1)?,
                )?;
            }
        }
        Ok(())
    }

    /// Scatter saved rows to new owners. Repeated parents are allowed; targets
    /// must be unique. All sources come from the snapshot, so permutations and
    /// one-to-many forks cannot overwrite a still-needed parent.
    pub fn fork_on(
        &self,
        layers: &mut [LinearLayerState<T, D>],
        parents: &[usize],
        targets: &[usize],
        scope: &D::Scope,
    ) -> OpResult<()> {
        if parents.len() != targets.len()
            || layers.len() != self.layers.len()
            || parents.iter().any(|&p| p >= self.slots.len())
            || targets
                .iter()
                .enumerate()
                .any(|(i, s)| targets[..i].contains(s))
            || layers.iter().zip(&self.layers).any(|(dst, src)| {
                dst.dims != src.dims
                    || targets.iter().any(|&s| s >= dst.conv.shape()[0])
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(dst.conv.device())
                    || infer_core::device::Device::device_id(src.conv.device())
                        != infer_core::device::Device::device_id(
                            infer_core::exec::ExecScope::device(scope),
                        )
            })
        {
            return Err(OpError::Shape("invalid recurrent fork mapping".into()));
        }
        for (dst, src) in layers.iter().zip(&self.layers) {
            for (&parent, &target) in parents.iter().zip(targets) {
                D::copy_tensor(
                    scope,
                    &src.conv.narrow(0, parent, 1)?,
                    &mut dst.conv.narrow(0, target, 1)?,
                )?;
                D::copy_tensor(
                    scope,
                    &src.ssm.narrow(0, parent, 1)?,
                    &mut dst.ssm.narrow(0, target, 1)?,
                )?;
            }
        }
        Ok(())
    }

    pub fn capture(layers: &[LinearLayerState<T, D>], slots: &[usize]) -> OpResult<Self> {
        let unique: HashSet<_> = slots.iter().copied().collect();
        if slots.is_empty()
            || unique.len() != slots.len()
            || layers
                .iter()
                .any(|l| slots.iter().any(|&s| s >= l.conv.shape()[0]))
        {
            return Err(OpError::Shape("invalid recurrent snapshot slots".into()));
        }
        let mut saved = Vec::with_capacity(layers.len());
        for layer in layers {
            let copy = LinearLayerState::new(layer.dims, slots.len(), layer.conv.device())?;
            for (row, &slot) in slots.iter().enumerate() {
                copy.conv
                    .narrow(0, row, 1)?
                    .copy_from(&layer.conv.narrow(0, slot, 1)?)?;
                copy.ssm
                    .narrow(0, row, 1)?
                    .copy_from(&layer.ssm.narrow(0, slot, 1)?)?;
            }
            saved.push(copy);
        }
        Ok(Self {
            slots: slots.to_vec(),
            layers: saved,
        })
    }

    pub fn restore(&self, layers: &mut [LinearLayerState<T, D>]) -> OpResult<()> {
        // Validate every destination before changing any layer.
        if layers.len() != self.layers.len()
            || layers.iter().zip(&self.layers).any(|(dst, src)| {
                dst.dims != src.dims
                    || self.slots.iter().any(|&s| s >= dst.conv.shape()[0])
                    || infer_core::device::Device::device_id(dst.conv.device())
                        != infer_core::device::Device::device_id(src.conv.device())
            })
        {
            return Err(OpError::Shape("recurrent snapshot layout mismatch".into()));
        }
        for (dst, src) in layers.iter_mut().zip(&self.layers) {
            for (row, &slot) in self.slots.iter().enumerate() {
                dst.conv
                    .narrow(0, slot, 1)?
                    .copy_from(&src.conv.narrow(0, row, 1)?)?;
                dst.ssm
                    .narrow(0, slot, 1)?
                    .copy_from(&src.ssm.narrow(0, row, 1)?)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn snapshot_is_owned_and_restores_only_selected_slots() {
        let dims = LinearDims {
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            conv_kernel_dim: 4,
        };
        let mut layers = vec![LinearLayerState::<f32, _>::new(dims, 3, &Cpu).unwrap()];
        let fill = |state: &LinearLayerState<f32, Cpu>, value| {
            for tensor in [&state.conv, &state.ssm] {
                tensor
                    .clone()
                    .upload_from_host(&vec![value; tensor.numel()])
                    .unwrap();
            }
        };
        fill(&layers[0], 2.0);
        let snapshot = LinearSnapshot::capture(&layers, &[2, 0]).unwrap();
        fill(&layers[0], 7.0);
        snapshot.restore(&mut layers).unwrap();
        for t in [&layers[0].conv, &layers[0].ssm] {
            for (slot, expected) in [(0, 2.0), (1, 7.0), (2, 2.0)] {
                assert!(
                    t.narrow(0, slot, 1)
                        .unwrap()
                        .to_host_vec()
                        .unwrap()
                        .iter()
                        .all(|&x| x == expected)
                );
            }
        }
        assert!(LinearSnapshot::capture(&layers, &[0, 0]).is_err());
        assert!(LinearSnapshot::capture(&layers, &[3]).is_err());
        // Repeated restore does not consume or alias the checkpoint.
        fill(&layers[0], 9.0);
        snapshot.restore(&mut layers).unwrap();
        assert_eq!(layers[0].ssm.to_host_vec().unwrap()[0], 2.0);
    }
    #[test]
    fn reserved_snapshot_reuses_storage_with_reordered_and_smaller_batches() {
        let dims = LinearDims {
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            conv_kernel_dim: 4,
        };
        let mut layers = vec![LinearLayerState::<f32, _>::new(dims, 3, &Cpu).unwrap()];
        let scope = infer_core::exec::HostScope::new(Cpu);
        let mut saved = LinearSnapshot::reserve(&layers, 3).unwrap();
        let address = saved.layers[0].ssm.data_ptr();
        for slots in [&[2, 0][..], &[1][..], &[0, 1, 2][..]] {
            for t in [&layers[0].conv, &layers[0].ssm] {
                for slot in 0..3 {
                    let mut row = t.narrow(0, slot, 1).unwrap();
                    row.upload_from_host(&vec![slot as f32 + 1.; row.numel()])
                        .unwrap();
                }
            }
            saved.capture_on(&layers, slots, &scope).unwrap();
            for t in [&layers[0].conv, &layers[0].ssm] {
                t.clone().upload_from_host(&vec![9.; t.numel()]).unwrap();
            }
            saved.restore_on(&mut layers, &scope).unwrap();
            for t in [&layers[0].conv, &layers[0].ssm] {
                for slot in 0..3 {
                    let want = if slots.contains(&slot) {
                        slot as f32 + 1.
                    } else {
                        9.
                    };
                    assert!(
                        t.narrow(0, slot, 1)
                            .unwrap()
                            .to_host_vec()
                            .unwrap()
                            .iter()
                            .all(|&x| x == want)
                    );
                }
            }
            assert_eq!(saved.layers[0].ssm.data_ptr(), address);
        }
        assert!(saved.capture_on(&layers, &[0, 0], &scope).is_err());
        assert!(saved.capture_on(&layers, &[3], &scope).is_err());
    }
}
