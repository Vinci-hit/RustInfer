//! Lightweight multi-shard safetensors loader for diffusers-format models
//! (DiT transformer, VAE, etc.) where `config.json` is NOT the LLM
//! `ModelFileConfig` format.
//!
//! Unlike `ModelLoader`, this loader only handles weight loading and does not
//! parse the config. Each model (DiT, VAE) parses its own config.json directly.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::TensorView;

use crate::base::error::{Error, Result};
use crate::model::common::safetensor_loader::{SafetensorReader, load_and_mmap};

/// Multi-shard safetensors loader without config coupling.
pub struct DiffusersLoader {
    _mmaps: HashMap<PathBuf, Mmap>,
    tensor_to_file: HashMap<String, PathBuf>,
    readers: HashMap<PathBuf, SafetensorReader<'static>>,
}

impl DiffusersLoader {
    /// Load from a directory.
    ///
    /// Supports:
    /// - Single-file: `{model_name}.safetensors` (any name) — if `model_dir`
    ///   contains exactly one `*.safetensors` file and no `*.index.json`.
    /// - Sharded with index: `*.safetensors.index.json` + referenced shard files.
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find index file
        let mut tensor_to_file: HashMap<String, PathBuf> = HashMap::new();
        let mut files_to_mmap: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

        // Look for any .safetensors.index.json
        let mut index_path: Option<PathBuf> = None;
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".safetensors.index.json") {
                    index_path = Some(path);
                    break;
                }
            }
        }

        if let Some(idx_path) = index_path {
            let file = File::open(&idx_path)?;
            let index: serde_json::Value = serde_json::from_reader(file)
                .map_err(|e| Error::InvalidArgument(format!("Failed to parse index.json: {}", e)))?;
            let weight_map = index["weight_map"].as_object().ok_or_else(|| {
                Error::InvalidArgument("weight_map not found in index.json".to_string())
            })?;
            for (name, filename) in weight_map {
                if let Some(fname) = filename.as_str() {
                    let fpath = model_dir.join(fname);
                    tensor_to_file.insert(name.clone(), fpath.clone());
                    files_to_mmap.insert(fpath);
                }
            }
        } else {
            // Find a single .safetensors file
            let mut single: Option<PathBuf> = None;
            for entry in std::fs::read_dir(model_dir)? {
                let entry = entry?;
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.ends_with(".safetensors") {
                        if single.is_some() {
                            return Err(Error::InvalidArgument(format!(
                                "Multiple .safetensors files but no index.json in {:?}", model_dir
                            )).into());
                        }
                        single = Some(path);
                    }
                }
            }
            let single = single.ok_or_else(|| {
                Error::InvalidArgument(format!("No .safetensors file found in {:?}", model_dir))
            })?;
            // mmap once to enumerate tensor names
            let mmap = load_and_mmap(&single)?;
            let reader = SafetensorReader::new(&mmap)?;
            for name in reader.get_tensor_names() {
                tensor_to_file.insert(name, single.clone());
            }
            files_to_mmap.insert(single);
        }

        // mmap all referenced files
        let mut mmaps: HashMap<PathBuf, Mmap> = HashMap::new();
        for file_path in files_to_mmap {
            let mmap = load_and_mmap(&file_path)?;
            mmaps.insert(file_path, mmap);
        }

        // Build readers with 'static lifetime (same trick as ModelLoader)
        let mut readers: HashMap<PathBuf, SafetensorReader<'static>> = HashMap::new();
        for (path, mmap) in &mmaps {
            let mmap_static: &'static [u8] = unsafe { std::mem::transmute(mmap.as_ref()) };
            let reader = SafetensorReader::new(mmap_static)?;
            readers.insert(path.clone(), reader);
        }

        Ok(Self {
            _mmaps: mmaps,
            tensor_to_file,
            readers,
        })
    }

    /// Get a tensor view by name.
    pub fn get_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let file_path = self.tensor_to_file.get(name).ok_or_else(|| {
            Error::InvalidArgument(format!("Tensor '{}' not found in loader index", name))
        })?;
        let reader = self.readers.get(file_path).ok_or_else(|| {
            Error::InternalError(format!("No reader for file {:?}", file_path))
        })?;
        reader.get_tensor(name)
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_file.contains_key(name)
    }

    /// Iterate over all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_file.keys().map(|s| s.as_str()).collect()
    }
}
