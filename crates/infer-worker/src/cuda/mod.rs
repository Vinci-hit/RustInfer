pub mod ffi;
pub mod error;
pub mod device;
pub mod config;
pub use config::CudaConfig;
pub use config::FLASH_DECODE_N_SPLIT;