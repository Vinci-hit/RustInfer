//! Safetensors reader — `infra/io` adapter.
//!
//! Single point of filesystem access for model weights. Supports both
//! single-file `model.safetensors` and HuggingFace-style sharded layouts
//! (`model.safetensors.index.json` + `model-XXXXX-of-YYYYY.safetensors`).
//!
//! ## DDD positioning
//! - **infra** layer: depends on OS (mmap) and the `safetensors` crate.
//! - The domain doesn't know files exist; loaders take a
//!   `&SafetensorsReader` rather than a path, isolating I/O.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

/// One shard: an mmap'd file + its parsed `SafeTensors` header.
struct Shard {
    // SAFETY: `_mmap` outlives `header` (declared first → dropped second).
    header: SafeTensors<'static>,
    _mmap: Box<Mmap>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("open {}: {}", path.display(), e))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {}", path.display(), e))?;
        let mmap = Box::new(mmap);
        // SAFETY: Box<Mmap> stays in self; pointer is stable.
        let bytes: &'static [u8] = unsafe {
            std::slice::from_raw_parts(mmap.as_ptr(), mmap.len())
        };
        let header = SafeTensors::deserialize(bytes)
            .map_err(|e| format!("safetensors deserialize {}: {}", path.display(), e))?;
        Ok(Self { header, _mmap: mmap })
    }
}

/// Owns one or more mmap'd safetensors files and routes `read_view` calls
/// to the right shard via the optional `weight_map` from
/// `model.safetensors.index.json`.
pub struct SafetensorsReader {
    shards: Vec<Shard>,
    /// Tensor name → shard index. For single-file readers this is empty
    /// and we fall back to scanning shard 0.
    name_to_shard: HashMap<String, usize>,
}

impl SafetensorsReader {
    /// Open a model directory or a single `.safetensors` file.
    ///
    /// - If `path` is a `.safetensors` file: load it as a single shard.
    /// - If `path` is a directory:
    ///     - first try `*.safetensors.index.json` (any prefix — `model.`,
    ///       `diffusion_pytorch_model.`, etc.) for a sharded layout
    ///     - else fall back to a single `*.safetensors` file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let meta = std::fs::metadata(path)
            .map_err(|e| format!("stat {}: {}", path.display(), e))?;
        if meta.is_file() {
            return Self::open_single(path);
        }
        if meta.is_dir() {
            // Look for any *.safetensors.index.json first.
            let entries = std::fs::read_dir(path)
                .map_err(|e| format!("read_dir {}: {}", path.display(), e))?;
            let mut index_path: Option<PathBuf> = None;
            let mut single_path: Option<PathBuf> = None;
            for e in entries {
                let e = e.map_err(|err| format!("dirent: {}", err))?;
                let p = e.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name.ends_with(".safetensors.index.json") {
                        index_path = Some(p);
                    } else if name.ends_with(".safetensors") {
                        // Track single-file fallback (only used if no index).
                        single_path = Some(p);
                    }
                }
            }
            if let Some(idx) = index_path {
                return Self::open_sharded(path, &idx);
            }
            if let Some(single) = single_path {
                return Self::open_single(&single);
            }
            return Err(format!(
                "no *.safetensors.index.json or *.safetensors under {}",
                path.display(),
            ));
        }
        Err(format!("path is neither file nor directory: {}", path.display()))
    }

    fn open_single(path: &Path) -> Result<Self, String> {
        let shard = Shard::open(path)?;
        Ok(Self {
            shards: vec![shard],
            name_to_shard: HashMap::new(),
        })
    }

    fn open_sharded(model_dir: &Path, index_path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(index_path)
            .map_err(|e| format!("read {}: {}", index_path.display(), e))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse {}: {}", index_path.display(), e))?;
        let weight_map = value
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("{}: missing weight_map", index_path.display()))?;

        // Collect distinct shard filenames in the order they appear.
        let mut file_idx: HashMap<String, usize> = HashMap::new();
        let mut files: Vec<PathBuf> = Vec::new();
        for (_, file_val) in weight_map.iter() {
            let fname = file_val
                .as_str()
                .ok_or_else(|| "weight_map entry is not a string".to_string())?;
            if !file_idx.contains_key(fname) {
                file_idx.insert(fname.to_string(), files.len());
                files.push(model_dir.join(fname));
            }
        }

        // Open all shards.
        let mut shards: Vec<Shard> = Vec::with_capacity(files.len());
        for f in &files {
            shards.push(Shard::open(f)?);
        }

        // Build name_to_shard map.
        let mut name_to_shard: HashMap<String, usize> =
            HashMap::with_capacity(weight_map.len());
        for (name, file_val) in weight_map.iter() {
            let fname = file_val.as_str().unwrap();
            let idx = file_idx[fname];
            name_to_shard.insert(name.clone(), idx);
        }

        Ok(Self { shards, name_to_shard })
    }

    /// Borrow a tensor view by name (zero copy into the mmap).
    pub fn read_view(&self, name: &str) -> Result<TensorView<'_>, String> {
        if !self.name_to_shard.is_empty() {
            let idx = *self.name_to_shard.get(name)
                .ok_or_else(|| format!("tensor '{}' not in weight_map", name))?;
            self.shards[idx]
                .header
                .tensor(name)
                .map_err(|e| format!("tensor '{}' not in shard {}: {}", name, idx, e))
        } else {
            // Single-file: scan shard 0.
            self.shards[0]
                .header
                .tensor(name)
                .map_err(|e| format!("tensor '{}' not found: {}", name, e))
        }
    }

    /// Whether a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        if !self.name_to_shard.is_empty() {
            self.name_to_shard.contains_key(name)
        } else {
            self.shards[0].header.tensor(name).is_ok()
        }
    }

    /// Iterate every tensor name across all shards.
    pub fn names(&self) -> Vec<String> {
        if !self.name_to_shard.is_empty() {
            let mut v: Vec<String> = self.name_to_shard.keys().cloned().collect();
            v.sort();
            v
        } else {
            self.shards[0]
                .header
                .tensors()
                .into_iter()
                .map(|(n, _)| n.to_string())
                .collect()
        }
    }
}
