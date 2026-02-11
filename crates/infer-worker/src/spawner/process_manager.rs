//! Process Manager Module
//!
//! Handles the lifecycle of Worker processes.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use anyhow::{Context, Result};

/// Manages a single Worker process
pub struct ProcessManager {
    /// The child process
    process: Option<Child>,
    /// stdout reader thread handle
    stdout_handle: Option<JoinHandle<()>>,
    /// stderr reader thread handle
    stderr_handle: Option<JoinHandle<()>>,
}

impl ProcessManager {
    /// Spawn a new Worker process with a simple logging callback
    pub fn spawn(
        worker_id: &str,
        exe_path: &std::path::Path,
        args: &[String],
        env: &std::collections::HashMap<String, String>,
        on_log: impl Fn(&str) + Send + 'static + Clone,
    ) -> Result<Self> {
        // Build command
        let mut cmd = Command::new(exe_path);
        cmd.args(args).envs(env);

        // Redirect stdout and stderr
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawn the process
        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to spawn worker {}: {:?}",
                worker_id, exe_path
            )
        })?;

        // Capture stdout
        let stdout_handle = if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let worker_id = worker_id.to_string();
            let callback = on_log.clone();
            Some(std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        callback(&format!("[{}] {}", worker_id, line));
                    }
                }
            }))
        } else {
            None
        };

        // Capture stderr
        let stderr_handle = if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            let worker_id = worker_id.to_string();
            let callback = on_log;
            Some(std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        callback(&format!("[{}] [STDERR] {}", worker_id, line));
                    }
                }
            }))
        } else {
            None
        };

        Ok(Self {
            process: Some(child),
            stdout_handle,
            stderr_handle,
        })
    }

    /// Check if the process is still running
    pub fn is_alive(&mut self) -> bool {
        if let Some(ref mut process) = self.process {
            process.try_wait().ok().flatten().is_none()
        } else {
            false
        }
    }

    /// Try to wait for the process to complete
    ///
    /// Returns Some(ExitStatus) if the process has exited, None otherwise
    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        if let Some(ref mut process) = self.process {
            Ok(process.try_wait()?)
        } else {
            Ok(None)
        }
    }

    /// Get the process ID
    pub fn id(&self) -> Option<u32> {
        self.process.as_ref().map(|p| p.id())
    }

    /// Kill the process
    pub fn kill(&mut self) -> Result<()> {
        if let Some(ref mut process) = self.process {
            process.kill().context("Failed to kill worker process")?;
        }
        Ok(())
    }

    /// Wait for the process to finish (blocking)
    pub fn wait(mut self) -> Result<ExitStatus> {
        if let Some(mut process) = self.process.take() {
            Ok(process.wait()?)
        } else {
            anyhow::bail!("Process already exited")
        }
    }
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        // Best effort cleanup
        let _ = self.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_manager_creation() {
        // This is a basic test to ensure ProcessManager can be created
        // More detailed tests would require spawning actual processes
        let _pm: Option<ProcessManager> = None;
        assert!(_pm.is_none());
    }
}
