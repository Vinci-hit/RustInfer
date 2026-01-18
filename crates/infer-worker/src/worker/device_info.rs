//! Device Information Module
//!
//! Provides device-level information including GPU memory, utilization, etc.

use crate::base::DeviceType;

/// Device information structure
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type (CPU or CUDA with device ID)
    pub device_type: DeviceType,
    /// Device ID (0 for CPU, GPU index for CUDA)
    pub device_id: u32,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Free memory in bytes (at query time)
    pub free_memory: u64,
    /// Used memory in bytes (at query time)
    pub used_memory: u64,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub device_name: String,
}

impl DeviceInfo {
    /// Create DeviceInfo for a CPU device
    pub fn cpu() -> Self {
        // Get system memory info
        let (total, free) = get_system_memory();

        Self {
            device_type: DeviceType::Cpu,
            device_id: 0,
            total_memory: total,
            free_memory: free,
            used_memory: total.saturating_sub(free),
            device_name: "CPU".to_string(),
        }
    }

    /// Create DeviceInfo for a CUDA device
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: i32) -> anyhow::Result<Self> {
        use crate::cuda::{device, ffi};

        // Set device context
        device::set_current_device(device_id)?;

        // Get memory info
        let (free, total) = unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            let status = ffi::cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize);
            if status != 0 {
                anyhow::bail!("cudaMemGetInfo failed with error code: {}", status);
            }
            (free as u64, total as u64)
        };

        Ok(Self {
            device_type: DeviceType::Cuda(device_id),
            device_id: device_id as u32,
            total_memory: total,
            free_memory: free,
            used_memory: total.saturating_sub(free),
            device_name: format!("CUDA Device {}", device_id),
        })
    }

    /// Refresh memory statistics
    #[cfg(feature = "cuda")]
    pub fn refresh_memory(&mut self) -> anyhow::Result<()> {
        match self.device_type {
            DeviceType::Cpu => {
                let (total, free) = get_system_memory();
                self.total_memory = total;
                self.free_memory = free;
                self.used_memory = total.saturating_sub(free);
            }
            DeviceType::Cuda(device_id) => {
                use crate::cuda::{device, ffi};
                device::set_current_device(device_id)?;

                let (free, total) = unsafe {
                    let mut free: usize = 0;
                    let mut total: usize = 0;
                    let status = ffi::cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize);
                    if status != 0 {
                        anyhow::bail!("cudaMemGetInfo failed with error code: {}", status);
                    }
                    (free as u64, total as u64)
                };

                self.total_memory = total;
                self.free_memory = free;
                self.used_memory = total.saturating_sub(free);
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn refresh_memory(&mut self) -> anyhow::Result<()> {
        let (total, free) = get_system_memory();
        self.total_memory = total;
        self.free_memory = free;
        self.used_memory = total.saturating_sub(free);
        Ok(())
    }

    /// Get memory utilization as a fraction (0.0 - 1.0)
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            self.used_memory as f64 / self.total_memory as f64
        }
    }

    /// Get total memory in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get free memory in GB
    pub fn free_memory_gb(&self) -> f64 {
        self.free_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get used memory in GB
    pub fn used_memory_gb(&self) -> f64 {
        self.used_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self.device_type, DeviceType::Cuda(_))
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Cpu)
    }
}

/// Get system memory information (total, free) in bytes
fn get_system_memory() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        // Parse /proc/meminfo for Linux
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total = 0u64;
            let mut available = 0u64;

            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb) = parse_meminfo_value(line) {
                        total = kb * 1024;
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(kb) = parse_meminfo_value(line) {
                        available = kb * 1024;
                    }
                }
            }

            return (total, available);
        }
    }

    // Fallback: return default values
    (16 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024) // 16GB total, 8GB free
}

/// Parse a value from /proc/meminfo line (e.g., "MemTotal:       16000000 kB")
#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> Option<u64> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].parse().ok()
    } else {
        None
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}): {:.2} GB total, {:.2} GB free ({:.1}% used)",
            self.device_name,
            match self.device_type {
                DeviceType::Cpu => "CPU".to_string(),
                DeviceType::Cuda(id) => format!("CUDA:{}", id),
            },
            self.total_memory_gb(),
            self.free_memory_gb(),
            self.memory_utilization() * 100.0
        )
    }
}
