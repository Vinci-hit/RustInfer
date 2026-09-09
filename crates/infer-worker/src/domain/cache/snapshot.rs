//! Owned recurrent checkpoints. KV suffix retention is a separate concern.
use super::*;

/// Copies only the participating request slots, including convolution history
/// and FP32 delta-rule state. Source and checkpoint never share storage.
/// The caller must order execution around these default-device-stream copies.
pub struct LinearSnapshot<T: Dtype, D: LlmBackend> {
    slots: Vec<usize>,
    layers: Vec<LinearLayerState<T, D>>,
}

impl<T: Dtype, D: LlmBackend> LinearSnapshot<T, D> {
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
}
