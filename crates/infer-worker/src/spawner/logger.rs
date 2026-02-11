use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use std::collections::HashMap;

/// A single Worker's log writer
pub struct WorkerLogger {
    worker_id: String,
    device_id: u32,
    file: Option<Arc<Mutex<File>>>,
}

impl Clone for WorkerLogger {
    fn clone(&self) -> Self {
        Self {
            worker_id: self.worker_id.clone(),
            device_id: self.device_id,
            file: self.file.clone(),
        }
    }
}

impl WorkerLogger {
    /// Create a new WorkerLogger
    fn new(worker_id: String, device_id: u32, file: Option<Arc<Mutex<File>>>) -> Self {
        Self {
            worker_id,
            device_id,
            file,
        }
    }

    /// Write a log line
    pub fn write(&self, level: &str, message: &str) -> Result<()> {
        let formatted = format!(
            "[{}] [{}] [device:{}] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            level,
            self.device_id,
            message
        );

        // Write to file if available
        if let Some(ref file) = self.file {
            let mut f = file.lock().unwrap();
            writeln!(f, "{}", formatted)?;
            let _ = f.flush();
        }

        // Also log to stdout
        println!("{}", formatted);

        Ok(())
    }
}

/// Aggregates logs from multiple Workers (thread-safe)
#[derive(Clone)]
pub struct AggregateLogger {
    loggers: Arc<Mutex<HashMap<String, WorkerLogger>>>,
    log_dir: Option<std::path::PathBuf>,
}

impl AggregateLogger {
    /// Create a new AggregateLogger
    pub fn new(log_dir: Option<&Path>) -> Result<Self> {
        let log_dir = if let Some(dir) = log_dir {
            std::fs::create_dir_all(dir)?;
            Some(dir.to_path_buf())
        } else {
            None
        };

        Ok(Self {
            loggers: Arc::new(Mutex::new(HashMap::new())),
            log_dir,
        })
    }

    /// Register a Worker logger
    pub fn register_worker(&self, worker_id: &str, device_id: u32) -> Result<()> {
        let file = if let Some(ref log_dir) = self.log_dir {
            let log_path = log_dir.join(format!("{}-gpu{}.log", worker_id, device_id));
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)?;
            Some(Arc::new(Mutex::new(file)))
        } else {
            None
        };

        let logger = WorkerLogger::new(worker_id.to_string(), device_id, file);
        self.loggers.lock().unwrap().insert(worker_id.to_string(), logger);

        Ok(())
    }

    /// Log a message from a Worker
    pub fn log(&self, worker_id: &str, level: &str, message: &str) -> Result<()> {
        let loggers = self.loggers.lock().unwrap();
        if let Some(logger) = loggers.get(worker_id) {
            logger.write(level, message)
        } else {
            anyhow::bail!("Worker {} not registered", worker_id)
        }
    }

    /// Log info level
    pub fn info(&self, worker_id: &str, message: &str) -> Result<()> {
        self.log(worker_id, "INFO", message)
    }

    /// Log warn level
    pub fn warn(&self, worker_id: &str, message: &str) -> Result<()> {
        self.log(worker_id, "WARN", message)
    }

    /// Log error level
    pub fn error(&self, worker_id: &str, message: &str) -> Result<()> {
        self.log(worker_id, "ERROR", message)
    }

    /// Get a list of all registered workers
    pub fn worker_ids(&self) -> Vec<String> {
        self.loggers.lock().unwrap().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_logger_creation() {
        let logger = AggregateLogger::new(None).unwrap();
        assert_eq!(logger.worker_ids().len(), 0);
    }

    #[test]
    fn test_register_worker() {
        let logger = AggregateLogger::new(None).unwrap();
        logger.register_worker("worker-0", 0).unwrap();
        assert_eq!(logger.worker_ids().len(), 1);
    }
}
