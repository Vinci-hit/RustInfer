//! Two owned layer buffers: read N+1 on a producer while the caller uploads N.
use super::SafetensorsReader;
use infer_core::device::{HostBuffer, MemoryPort};
use infer_core::error::{OpError, OpResult};
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

struct TensorRange {
    file: Arc<File>,
    file_offset: u64,
    offset: usize,
    len: usize,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}
struct LayerPlan {
    tensors: HashMap<String, TensorRange>,
    bytes: usize,
}

pub struct PrefetchedLayer {
    plan: LayerPlan,
    buffer: Box<dyn HostBuffer>,
}
impl PrefetchedLayer {
    pub fn read_view(&self, name: &str) -> Option<Result<TensorView<'_>, String>> {
        self.plan.tensors.get(name).map(|entry| {
            TensorView::new(
                entry.dtype,
                entry.shape.clone(),
                &self.buffer.bytes()[entry.offset..entry.offset + entry.len],
            )
            .map_err(|error| format!("tensor '{name}': {error}"))
        })
    }
}

pub struct LayerPrefetch {
    ready: Option<mpsc::Receiver<OpResult<PrefetchedLayer>>>,
    free: Option<mpsc::SyncSender<Box<dyn HostBuffer>>>,
    worker: Option<JoinHandle<()>>,
}
impl LayerPrefetch {
    pub fn next_layer(&self) -> OpResult<PrefetchedLayer> {
        self.ready
            .as_ref()
            .expect("live prefetch")
            .recv()
            .map_err(|_| {
                OpError::Kernel("layer reader stopped before completing the model".into())
            })?
    }
    pub fn recycle(&self, layer: PrefetchedLayer) {
        // Receiver may already have exited after reading the last layer.
        let _ = self
            .free
            .as_ref()
            .expect("live prefetch")
            .send(layer.buffer);
    }
}
impl Drop for LayerPrefetch {
    fn drop(&mut self) {
        // Disconnect both directions before joining: the reader may be blocked
        // on an empty free queue or a full ready queue after a build error.
        self.ready.take();
        self.free.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
impl SafetensorsReader {
    /// Prefixes come from the model, e.g. `model.layers.0.`. Only metadata is
    /// inspected here; payload bytes are read directly into the two buffers.
    pub fn prefetch_layers<D: MemoryPort>(
        &self,
        prefixes: impl IntoIterator<Item = String>,
        device: &D,
    ) -> OpResult<LayerPrefetch> {
        let names = self.names();
        let mut plans = Vec::new();
        for prefix in prefixes {
            let mut plan = LayerPlan {
                tensors: HashMap::new(),
                bytes: 0,
            };
            for name in names.iter().filter(|name| name.starts_with(&prefix)) {
                let shard_index = self.name_to_shard.get(name).copied().unwrap_or(0);
                let shard = &self.shards[shard_index];
                let view = shard
                    .header
                    .tensor(name)
                    .map_err(|e| OpError::Kernel(e.to_string()))?;
                let offset = plan.bytes;
                plan.bytes = plan
                    .bytes
                    .checked_add(view.data().len())
                    .ok_or_else(|| OpError::Shape("layer read buffer size overflows".into()))?;
                plan.tensors.insert(
                    name.clone(),
                    TensorRange {
                        file: shard.file.clone(),
                        file_offset: (view.data().as_ptr() as usize - shard._mmap.as_ptr() as usize)
                            as u64,
                        offset,
                        len: view.data().len(),
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                    },
                );
            }
            plans.push(plan);
        }
        let capacity = plans.iter().map(|plan| plan.bytes).max().unwrap_or(0);
        let (free_tx, free_rx) = mpsc::sync_channel(2);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        if !plans.is_empty() {
            // Allocate once, sized from the largest actual layer, not GPU RAM
            // or a configurable chunk size. No allocation on a layer handoff.
            for _ in 0..2 {
                free_tx
                    .send(device.alloc_host_buffer(capacity)?)
                    .map_err(|_| OpError::Kernel("initialize layer read buffers".into()))?;
            }
        }
        tracing::debug!(
            capacity_bytes = capacity,
            layers = plans.len(),
            "layer read/upload pipeline"
        );
        let worker = std::thread::Builder::new()
            .name("weight-reader".into())
            .spawn(move || {
                for plan in plans {
                    let Ok(mut buffer) = free_rx.recv() else {
                        break;
                    };
                    let read = (|| -> OpResult<()> {
                        // Offset ordering avoids jumping around within each shard.
                        let mut entries: Vec<_> = plan.tensors.iter().collect();
                        entries.sort_by_key(|(_, entry)| {
                            (Arc::as_ptr(&entry.file) as usize, entry.file_offset)
                        });
                        for (name, entry) in entries {
                            entry
                                .file
                                .read_exact_at(
                                    &mut buffer.bytes_mut()[entry.offset..entry.offset + entry.len],
                                    entry.file_offset,
                                )
                                .map_err(|error| {
                                    OpError::Kernel(format!("read tensor '{name}': {error}"))
                                })?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = read {
                        let _ = ready_tx.send(Err(error));
                        break;
                    }
                    if ready_tx.send(Ok(PrefetchedLayer { plan, buffer })).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| OpError::Kernel(format!("start weight reader: {error}")))?;
        Ok(LayerPrefetch {
            ready: Some(ready_rx),
            free: Some(free_tx),
            worker: Some(worker),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "rustinfer-layer-read-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            let mut weight_map = serde_json::Map::new();
            // Every layer spans two files; layer sizes vary and include odd tails.
            for shard in 0..2 {
                let payloads: Vec<Vec<u8>> = (0..4)
                    .map(|layer| vec![(layer * 2 + shard) as u8; layer + 3])
                    .collect();
                let views: Vec<_> = payloads
                    .iter()
                    .enumerate()
                    .map(|(layer, bytes)| {
                        let name = format!("model.layers.{layer}.part{shard}");
                        weight_map.insert(
                            name.clone(),
                            serde_json::json!(format!("part{shard}.safetensors")),
                        );
                        (
                            name,
                            TensorView::new(safetensors::Dtype::U8, vec![bytes.len()], bytes)
                                .unwrap(),
                        )
                    })
                    .collect();
                safetensors::tensor::serialize_to_file(
                    views,
                    None,
                    &path.join(format!("part{shard}.safetensors")),
                )
                .unwrap();
            }
            std::fs::write(
                path.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
            )
            .unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn pipeline(reader: &SafetensorsReader) -> LayerPrefetch {
        reader
            .prefetch_layers((0..4).map(|i| format!("model.layers.{i}.")), &Cpu)
            .unwrap()
    }
    fn check(layer: &PrefetchedLayer, index: usize) {
        for shard in 0..2 {
            let view = layer
                .read_view(&format!("model.layers.{index}.part{shard}"))
                .unwrap()
                .unwrap();
            assert_eq!(view.data(), vec![(index * 2 + shard) as u8; index + 3]);
        }
    }
    #[test]
    fn reads_next_layer_while_current_is_held_and_reuses_only_recycled_buffer() {
        let fixture = Fixture::new();
        let reader = SafetensorsReader::open(&fixture.0).unwrap();
        let pipeline = pipeline(&reader);
        drop(reader); // metadata/file handles suffice; producer never reads the mmap
        let first = pipeline.next_layer().unwrap();
        let first_ptr = first.buffer.bytes().as_ptr();
        let second = pipeline.next_layer().unwrap(); // no recycle: independent second buffer
        assert_ne!(first_ptr, second.buffer.bytes().as_ptr());
        assert_eq!(first.buffer.bytes().len(), 12); // largest layer: two 6-byte tensors
        check(&first, 0);
        check(&second, 1);
        assert!(matches!(
            pipeline.ready.as_ref().unwrap().try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
        pipeline.recycle(first);
        let third = pipeline.next_layer().unwrap();
        assert_eq!(first_ptr, third.buffer.bytes().as_ptr());
        check(&third, 2);
        check(&second, 1); // retained layer was not overwritten
        pipeline.recycle(second);
        let fourth = pipeline.next_layer().unwrap();
        check(&fourth, 3);
        pipeline.recycle(third);
        pipeline.recycle(fourth);
    }
    #[test]
    fn early_model_exit_disconnects_reader_even_with_a_full_ready_queue() {
        let fixture = Fixture::new();
        let reader = SafetensorsReader::open(&fixture.0).unwrap();
        let pipeline = pipeline(&reader);
        let held = pipeline.next_layer().unwrap();
        drop(pipeline);
        check(&held, 0);
    }
    #[test]
    fn file_read_failure_reaches_consumer_and_reader_can_join() {
        let fixture = Fixture::new();
        let mut reader = SafetensorsReader::open(&fixture.0).unwrap();
        // Replace only read descriptors, leaving valid mmap metadata intact.
        for shard in &mut reader.shards {
            shard.file = Arc::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(fixture.0.join("part0.safetensors"))
                    .unwrap(),
            );
        }
        let pipeline = pipeline(&reader);
        let error = pipeline
            .next_layer()
            .err()
            .expect("pread on write-only descriptor fails");
        assert!(error.to_string().contains("read tensor"));
        drop(pipeline);
    }
}
