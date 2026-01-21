//! Device Discovery Module
//!
//! Provides functionality to discover and filter available computing devices (CPU/GPU).

use anyhow::{Context, Result};
use std::fmt;

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub total_memory: u64,
    pub free_memory: u64,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
}

impl fmt::Display for CudaDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU {}: {} ({:.2}GB total, {:.2}GB free, SM_{}.{})",
            self.device_id,
            self.name,
            self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            self.compute_capability.0,
            self.compute_capability.1
        )
    }
}

/// Available device type (supports CPU and CUDA)
#[derive(Debug, Clone)]
pub enum AvailableDevice {
    Cpu,
    Cuda(CudaDeviceInfo),
}

impl fmt::Display for AvailableDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda(info) => write!(f, "{}", info),
        }
    }
}

impl AvailableDevice {
    /// Get device ID
    pub fn device_id(&self) -> i32 {
        match self {
            Self::Cpu => 0,
            Self::Cuda(info) => info.device_id,
        }
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Get total memory in bytes
    pub fn total_memory(&self) -> u64 {
        match self {
            Self::Cpu => get_system_memory().0,
            Self::Cuda(info) => info.total_memory,
        }
    }

    /// Get free memory in bytes
    pub fn free_memory(&self) -> u64 {
        match self {
            Self::Cpu => get_system_memory().1,
            Self::Cuda(info) => info.free_memory,
        }
    }
}

/// Device constraints for filtering
#[derive(Debug, Clone, Default)]
pub struct DeviceConstraints {
    /// Minimum available memory in MB
    pub min_memory_mb: Option<u64>,

    /// Exclude these device IDs
    pub exclude_device_ids: Vec<i32>,

    /// Only use these device IDs (if specified)
    pub include_device_ids: Option<Vec<i32>>,

    /// Maximum number of devices to use
    pub max_devices: Option<usize>,

    /// Include CPU as a device
    pub include_cpu: bool,
}

/// Device discovery module
pub struct DeviceDiscovery;

impl DeviceDiscovery {
    /// Discover all available CUDA devices
    ///
    /// This uses CUDA runtime API to query available GPUs.
    /// Returns an empty vector if CUDA is not available or no devices found.
    #[cfg(feature = "cuda")]
    pub fn discover_cuda_devices() -> Result<Vec<CudaDeviceInfo>> {
        use crate::cuda::{device, ffi};

        // Get device count - using the existing function from cuda module
        let device_count = unsafe {
            let mut count = 0i32;
            let status = ffi::cudaGetDeviceCount(&mut count);
            if status != 0 {
                return Ok(vec![]); // No CUDA devices or driver not available
            }
            count
        };

        if device_count == 0 {
            return Ok(vec![]);
        }

        let mut devices = Vec::new();

        for device_id in 0..device_count {
            // Set device context
            device::set_current_device(device_id)?;

            // Get device name - use a simpler approach for now
            let name = format!("GPU Device {}", device_id);

            // Get memory info
            let (free, total) = unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                let status = ffi::cudaMemGetInfo(&mut free, &mut total);
                if status != 0 {
                    return Err(anyhow::anyhow!(
                        "cudaMemGetInfo failed for device {}: error {}",
                        device_id,
                        status
                    ));
                }
                (free as u64, total as u64)
            };

            // Default values for compute capability and multiprocessor count
            // These will be filled properly when CUDA bindings are extended
            devices.push(CudaDeviceInfo {
                device_id,
                name,
                total_memory: total,
                free_memory: free,
                compute_capability: (0, 0), // TODO: Get from CUDA
                multiprocessor_count: 0, // TODO: Get from CUDA
            });
        }

        Ok(devices)
    }

    /// Discover all available CUDA devices
    #[cfg(not(feature = "cuda"))]
    pub fn discover_cuda_devices() -> Result<Vec<CudaDeviceInfo>> {
        Ok(vec![])
    }

    /// Discover all available devices (CPU + CUDA)
    pub fn discover_all() -> Result<Vec<AvailableDevice>> {
        let mut devices = Vec::new();

        // Add CPU if needed
        devices.push(AvailableDevice::Cpu);

        // Add CUDA devices
        let cuda_devices = Self::discover_cuda_devices()?;
        for cuda_device in cuda_devices {
            devices.push(AvailableDevice::Cuda(cuda_device));
        }

        Ok(devices)
    }

    /// Filter devices based on constraints
    pub fn filter_devices(
        devices: Vec<AvailableDevice>,
        constraints: &DeviceConstraints,
    ) -> Vec<AvailableDevice> {
        devices
            .into_iter()
            .filter(|device| {
                // Skip CPU if not requested
                if device.is_cpu() && !constraints.include_cpu {
                    return false;
                }

                // Check include list
                if let Some(ref include_ids) = constraints.include_device_ids {
                    if !include_ids.contains(&device.device_id()) {
                        return false;
                    }
                }

                // Check exclude list
                if constraints.exclude_device_ids.contains(&device.device_id()) {
                    return false;
                }

                // Check minimum memory
                if let Some(min_mb) = constraints.min_memory_mb {
                    let min_bytes = min_mb * 1024 * 1024;
                    if device.free_memory() < min_bytes {
                        return false;
                    }
                }

                true
            })
            .take(constraints.max_devices.unwrap_or(usize::MAX))
            .collect()
    }

    /// Print device summary
    pub fn print_summary(devices: &[AvailableDevice]) {
        println!("\n[Spawner] Device Discovery Summary:");
        println!("{}", "=".repeat(70));

        if devices.is_empty() {
            println!("⚠  No devices found!");
            return;
        }

        println!("Found {} device(s):", devices.len());
        println!();

        for device in devices {
            match device {
                AvailableDevice::Cpu => {
                    let (total, free) = get_system_memory();
                    println!("  ┌─ CPU Device");
                    println!("  │  Total Memory: {:.2} GB", total as f64 / (1024.0 * 1024.0 * 1024.0));
                    println!("  │  Free Memory:  {:.2} GB", free as f64 / (1024.0 * 1024.0 * 1024.0));
                    println!("  └────────────────────");
                }
                AvailableDevice::Cuda(info) => {
                    println!("  ┌─ GPU Device {}", info.device_id);
                    println!("  │  Name: {}", info.name);
                    println!("  │  Compute: {}.{}", info.compute_capability.0, info.compute_capability.1);
                    println!("  │  SM Count: {}", info.multiprocessor_count);
                    println!("  │  Total:  {:.2} GB", info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
                    println!("  │  Free:   {:.2} GB", info.free_memory as f64 / (1024.0 * 1024.0 * 1024.0));
                    println!("  └─────────────────────────────");
                }
            }
        }

        println!("{}", "=".repeat(70));
    }
}

/// Get system memory information (total, free) in bytes
fn get_system_memory() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
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

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            let total_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(total) = total_str.trim().parse::<u64>() {
                return (total, total / 2); // Rough estimate
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("wmic")
            .args(["OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/format:csv"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse the CSV output (skip header line)
            let mut lines = output_str.lines().skip(1);
            if let Some(line) = lines.next() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 3 {
                    if let (Ok(free_kb), Ok(total_kb)) = (
                        parts[1].trim().parse::<u64>(),
                        parts[2].trim().parse::<u64>(),
                    ) {
                        return (total_kb * 1024, free_kb * 1024);
                    }
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_constraints_default() {
        let constraints = DeviceConstraints::default();
        assert!(constraints.include_cpu == false);
        assert!(constraints.exclude_device_ids.is_empty());
        assert!(constraints.include_device_ids.is_none());
    }

    #[test]
    fn test_device_filtering() {
        let devices = vec![
            AvailableDevice::Cpu,
            AvailableDevice::Cuda(CudaDeviceInfo {
                device_id: 0,
                name: "GPU 0".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 8 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                multiprocessor_count: 108,
            }),
            AvailableDevice::Cuda(CudaDeviceInfo {
                device_id: 1,
                name: "GPU 1".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 8 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                multiprocessor_count: 108,
            }),
        ];

        let constraints = DeviceConstraints {
            include_cpu: false,
            include_device_ids: Some(vec![0]),
            ..Default::default()
        };

        let filtered = DeviceDiscovery::filter_devices(devices, &constraints);
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].is_cuda());
        assert_eq!(filtered[0].device_id(), 0);
    }

    #[test]
    fn test_available_device_display() {
        let device = AvailableDevice::Cpu;
        assert_eq!(format!("{}", device), "CPU");

        let cuda_device = AvailableDevice::Cuda(CudaDeviceInfo {
            device_id: 0,
            name: "RTX 4090".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024,
            free_memory: 20 * 1024 * 1024 * 1024,
            compute_capability: (8, 9),
            multiprocessor_count: 128,
        });
        let display = format!("{}", cuda_device);
        assert!(display.contains("GPU 0"));
        assert!(display.contains("RTX 4090"));
    }
}
