//! Process boundary — worker binary entry point + ZMQ communication + scheduling.
//!
//! Two-thread model:
//! - Runner thread: spins on SyncFlags, executes model.forward()
//! - SubScheduler thread: ZMQ communication + decode self-loop + batch assembly

pub mod sync_flags;
pub mod sub_scheduler;
pub mod control_pump;
pub mod data_pump;
pub mod serve_loop;
