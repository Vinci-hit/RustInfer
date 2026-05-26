//! Lock-free synchronization primitives for Runner ↔ SubScheduler handshake.
//!
//! Per step-buffer, two flags coordinate ownership:
//! - `input_ready`: SubScheduler signals Runner that input is ready
//! - `output_ready`: Runner signals SubScheduler that output is available

use std::sync::atomic::{AtomicBool, Ordering};

/// Per-buffer synchronization flags.
///
/// Memory ordering:
/// - Writer sets flag with `Release` (publishes the data written before)
/// - Reader loads with `Acquire` (sees all data written before the flag)
#[repr(align(64))] // cache line isolation
pub struct SyncFlags {
    /// SubScheduler → Runner: "workspace is filled, you can forward"
    pub input_ready: AtomicBool,
    /// Runner → SubScheduler: "output is ready, you can read"
    pub output_ready: AtomicBool,
    /// SubScheduler → Runner: "I've consumed the output, buffer is free"
    pub output_consumed: AtomicBool,
    /// Global shutdown signal
    pub shutdown: AtomicBool,
}

impl SyncFlags {
    pub const fn new() -> Self {
        Self {
            input_ready: AtomicBool::new(false),
            output_ready: AtomicBool::new(false),
            output_consumed: AtomicBool::new(true), // initially buffer is free
            shutdown: AtomicBool::new(false),
        }
    }

    // ─── SubScheduler side ───────────────────────────────────────────

    /// Signal that input is ready for this buffer.
    #[inline]
    pub fn signal_input_ready(&self) {
        self.input_ready.store(true, Ordering::Release);
    }

    /// Wait until output is ready (spin + yield).
    /// Returns false if shutdown was requested.
    pub fn wait_output_ready(&self) -> bool {
        loop {
            if self.shutdown.load(Ordering::Relaxed) { return false; }
            if self.output_ready.load(Ordering::Acquire) { return true; }
            std::hint::spin_loop();
        }
    }

    /// Acknowledge that output has been consumed.
    #[inline]
    pub fn signal_output_consumed(&self) {
        self.output_ready.store(false, Ordering::Relaxed);
        self.output_consumed.store(true, Ordering::Release);
    }

    // ─── Runner side ─────────────────────────────────────────────────

    /// Check if input is ready (non-blocking).
    #[inline]
    pub fn is_input_ready(&self) -> bool {
        self.input_ready.load(Ordering::Acquire)
    }

    /// Consume the input_ready signal.
    #[inline]
    pub fn consume_input(&self) {
        self.input_ready.store(false, Ordering::Relaxed);
    }

    /// Signal that output is ready.
    #[inline]
    pub fn signal_output_ready(&self) {
        self.output_ready.store(true, Ordering::Release);
    }

    /// Wait until output is consumed (buffer reusable).
    /// Returns false on shutdown.
    pub fn wait_output_consumed(&self) -> bool {
        loop {
            if self.shutdown.load(Ordering::Relaxed) { return false; }
            if self.output_consumed.load(Ordering::Acquire) { return true; }
            std::hint::spin_loop();
        }
    }

    /// Claim the buffer (clear output_consumed).
    #[inline]
    pub fn claim_buffer(&self) {
        self.output_consumed.store(false, Ordering::Relaxed);
    }

    // ─── Shutdown ────────────────────────────────────────────────────

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn handshake_single_step() {
        let flags = Arc::new(SyncFlags::new());
        let f2 = flags.clone();

        // Simulate runner thread
        let runner = thread::spawn(move || {
            // Wait for input
            while !f2.is_input_ready() { std::hint::spin_loop(); }
            f2.consume_input();
            // "Forward" (no-op)
            f2.signal_output_ready();
        });

        // SubScheduler side
        flags.signal_input_ready();
        assert!(flags.wait_output_ready());
        flags.signal_output_consumed();

        runner.join().unwrap();
    }

    #[test]
    fn shutdown_unblocks_wait() {
        let flags = Arc::new(SyncFlags::new());
        let f2 = flags.clone();

        let waiter = thread::spawn(move || {
            f2.wait_output_ready() // should return false
        });

        // Give waiter time to start spinning
        std::thread::sleep(std::time::Duration::from_millis(10));
        flags.request_shutdown();

        assert!(!waiter.join().unwrap());
    }
}
