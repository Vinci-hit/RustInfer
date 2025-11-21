use memmap2::MmapOptions;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

use crate::base::error::{Result, Error};

#[derive(Debug)]
pub struct SafetensorReader<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> SafetensorReader<'a> {
    /// 从一个已经映射到内存的缓冲区创建 Reader
    pub fn new(buffer: &'a [u8]) -> Result<Self> {
        // 将 safetensors 的错误手动映射到我们的统一 Error 类型
        let tensors = SafeTensors::deserialize(buffer)
            .map_err(|e| Error::InvalidArgument(format!("Failed to deserialize safetensors: {}", e)))?;

        Ok(Self { tensors })
    }

    /// 根据名称获取一个张量视图
    pub fn get_tensor(&self, name: &str) -> Result<TensorView<'a>> {
        self.tensors
            .tensor(name)
            .map_err(|e| Error::InvalidArgument(format!("Tensor '{}' not found in this file: {}", name, e)).into())
    }

    pub fn get_tensor_names(&self) -> Vec<String> {
        self.tensors.names().into_iter().map(|s| s.to_string()).collect()
    }

    pub fn print_tensor_info(&self, name: &str) {
        match self.get_tensor(name) {
            Ok(tensor_view) => {
                println!("Tensor: '{}'", name);
                println!("  - Dtype: {:?}", tensor_view.dtype());
                println!("  - Shape: {:?}", tensor_view.shape());
                let data = tensor_view.data();
                println!("  - Data size (bytes): {}", data.len());
                // 仅打印前几个元素以作演示
                if data.len() > 20 {
                    println!("  - Data (first 5 elements as bytes): {:?}", &data[..20]);
                } else {
                     println!("  - Data (bytes): {:?}", data);
                }
            }
            Err(e) => {
                eprintln!("Error getting tensor '{}': {}", name, e);
            }
        }
    }
}

/// 辅助函数，用于内存映射文件
pub fn load_and_mmap(path: &Path) -> Result<memmap2::Mmap> {
    let file = File::open(path)?; // `?` 会自动使用 #[from] Io
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use safetensors::Dtype;

    use super::*;
    #[test]
    fn safetensor_test() {
        let filename = "/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct/model.safetensors"; // 将此替换为你的模型文件路径

        // 检查文件是否存在
        if !Path::new(filename).exists() {
            panic!("Error: File '{}' not found. Please ensure the model file is in the correct path.", filename);
        }

        // 1. 通过内存映射加载文件
        eprintln!("Loading model from '{}'...", filename);
        let buffer = match load_and_mmap(Path::new(filename)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Failed to load and map file: {}", e);
                return;
            }
        };

        // 2. 创建 SafetensorReader 实例
        let reader = match SafetensorReader::new(&buffer) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to deserialize safetensors: {}", e);
                return;
            }
        };
        eprintln!("Model loaded successfully.");

        // 3. 获取并打印所有张量的名称
        let tensor_names = reader.get_tensor_names();
        eprintln!("\nAvailable tensors in the model:");
        for name in &tensor_names {
            eprintln!("- {}", name);
        }

        // 4. 获取并检查一个特定的张量
        if let Some(first_tensor_name) = tensor_names.get(0) {
            eprintln!("\nInspecting the first tensor: '{}'", first_tensor_name);
            reader.print_tensor_info(first_tensor_name);

            // 演示如何访问数据
            match reader.get_tensor(first_tensor_name) {
                Ok(tensor_view) => {
                    // 根据张量的数据类型（Dtype）来安全地转换和使用数据
                    // 例如，如果知道它是 F32 (f32) 类型
                    if tensor_view.dtype() == Dtype::BF16 {
                        // 注意：这是一个不安全的操作，因为需要确保字节对齐和类型正确
                        let data_bf16: &[bf16] = unsafe {
                            std::slice::from_raw_parts(
                                tensor_view.data().as_ptr() as *const bf16,
                                tensor_view.data().len() / std::mem::size_of::<bf16>(),
                            )
                        };
                        eprintln!("  - Successfully viewed data as &[BF16]. First 5 floats: {:?}", &data_bf16[..5.min(data_bf16.len())]);
                    }
                },
                Err(_) => {} // 已在 print_tensor_info 中处理
            }
        } else {
            eprintln!("\nNo tensors found in the file.");
        }
    }
}