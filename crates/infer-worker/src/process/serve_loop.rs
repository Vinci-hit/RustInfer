//! Serve loop — spawns Runner thread + SubScheduler thread, coordinates lifecycle.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::scheduler_to_worker_data::BatchCommand;
use infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput;

use crate::domain::ports::OpBackend;
use crate::domain::types::Dtype;
use crate::domain::model::LlmModel;
use crate::app::model_runner::ModelRunner;
use super::sync_flags::SyncFlags;
use super::sub_scheduler::SubScheduler;
use super::control_pump::ControlPump;
use super::data_pump::DataPump;

/// Worker serve configuration.
pub struct ServeConfig {
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    pub heartbeat_interval_ms: u64,
}

/// ModelRunner wrapped with SyncFlags for thread coordination.
pub struct ModelRunnerWithSync<T: Dtype, D: OpBackend, M: LlmModel<T, D>> {
    pub runner: std::cell::UnsafeCell<ModelRunner<T, D, M>>,
    pub sync: SyncFlags,
}

unsafe impl<T: Dtype, D: OpBackend, M: LlmModel<T, D>> Send for ModelRunnerWithSync<T, D, M> {}
unsafe impl<T: Dtype, D: OpBackend, M: LlmModel<T, D>> Sync for ModelRunnerWithSync<T, D, M> {}

impl<T: Dtype, D: OpBackend, M: LlmModel<T, D>> ModelRunnerWithSync<T, D, M> {
    pub fn new(runner: ModelRunner<T, D, M>) -> Self {
        Self {
            runner: std::cell::UnsafeCell::new(runner),
            sync: SyncFlags::new(),
        }
    }
}

/// The two-thread serving runtime.
pub fn run_serve_loop<T, D, M>(
    runner_sync: Arc<ModelRunnerWithSync<T, D, M>>,
    control: ControlPump,
    data: DataPump,
    config: ServeConfig,
) where
    T: Dtype + 'static,
    D: OpBackend + 'static,
    M: LlmModel<T, D> + Send + Sync + 'static,
{
    let runner_for_thread = runner_sync.clone();

    // ─── Runner thread ───
    let runner_thread = thread::Builder::new()
        .name("runner".to_string())
        .spawn(move || {
            runner_loop(&runner_for_thread);
        })
        .expect("failed to spawn runner thread");

    // ─── SubScheduler loop (runs on current thread) ───
    sub_scheduler_loop(runner_sync.clone(), control, data, &config);

    // Shutdown
    runner_sync.sync.request_shutdown();
    runner_thread.join().expect("runner panicked");
}

/// Runner thread: spin-waits for input, executes forward, signals output.
fn runner_loop<T: Dtype, D: OpBackend, M: LlmModel<T, D>>(
    shared: &ModelRunnerWithSync<T, D, M>,
) {
    loop {
        if shared.sync.is_shutdown() { break; }
        if !shared.sync.is_input_ready() {
            std::hint::spin_loop();
            continue;
        }
        shared.sync.consume_input();

        // Execute model forward
        // The SubScheduler has written input tensors into the runner's workspace;
        // the runner calls step() and writes output tokens.
        // (In full integration: runner.step() is called here with pre-written workspace)

        shared.sync.signal_output_ready();

        // Wait for SubScheduler to consume output
        if !shared.sync.wait_output_consumed() { break; }
        shared.sync.claim_buffer();
    }
}

/// SubScheduler loop: ZMQ communication + decode self-loop.
fn sub_scheduler_loop<T: Dtype, D: OpBackend, M: LlmModel<T, D>>(
    shared: Arc<ModelRunnerWithSync<T, D, M>>,
    control: ControlPump,
    data: DataPump,
    config: &ServeConfig,
) {
    let mut sched = SubScheduler::new(config.max_batch_tokens, config.max_batch_seqs);
    let heartbeat_interval = Duration::from_millis(config.heartbeat_interval_ms);
    let mut last_heartbeat = Instant::now();

    loop {
        // 1. Handle control plane (non-blocking)
        if let Ok(Some((msg, _req_id))) = control.try_recv(0) {
            match msg {
                SchedulerControlMessage::Shutdown => {
                    shared.sync.request_shutdown();
                    break;
                }
                SchedulerControlMessage::Cancel(cancel) => {
                    sched.cancel_sequence(cancel.sequence_id);
                }
                SchedulerControlMessage::Ping => {
                    let _ = control.send(
                        infer_protocol::worker_to_scheduler_control::WorkerControlMessage::Pong,
                        _req_id,
                    );
                }
                _ => {} // Drain, UnloadModel, etc. handled in production
            }
        }

        // 2. Receive new prefill commands (non-blocking, drain all available)
        while let Ok(Some(cmd)) = data.try_recv_batch(0) {
            match cmd {
                BatchCommand::Prefill(prefill) => {
                    sched.pending_prefills.push_back(prefill);
                }
                BatchCommand::DiffusionBatch(_diff) => {
                    // Diffusion: run synchronously (not interleaved with LLM decode)
                    let output = DiffusionBatchOutput { results: vec![] };
                    let _ = data.send_diffusion_output(&output);
                }
            }
        }

        // 3. Build next batch
        let batch = match sched.build_mixed_batch() {
            Some(b) => b,
            None => {
                // Idle — blocking wait for next command with heartbeat timeout
                maybe_heartbeat(&control, &sched, &mut last_heartbeat, heartbeat_interval);
                match data.try_recv_batch(heartbeat_interval.as_millis() as i64) {
                    Ok(Some(BatchCommand::Prefill(cmd))) => {
                        sched.pending_prefills.push_back(cmd);
                    }
                    Ok(Some(BatchCommand::DiffusionBatch(_diff))) => {
                        let output = DiffusionBatchOutput { results: vec![] };
                        let _ = data.send_diffusion_output(&output);
                    }
                    _ => {}
                }
                if shared.sync.is_shutdown() { break; }
                continue;
            }
        };

        let num_items = batch.num_decode + batch.num_prefill;
        sched.last_batch = Some(batch);

        // 4. Write workspace → signal Runner
        // (Full integration: write batch.input_tokens into runner workspace device tensors)
        shared.sync.signal_input_ready();

        // 5. Wait for Runner output
        if !shared.sync.wait_output_ready() { break; }

        // 6. Read output tokens from Runner
        // (Full integration: D2H copy from runner's output_tokens tensor)
        // For now: use argmax results that Runner computed
        let output_tokens = vec![0i32; num_items]; // placeholder until workspace integration

        // 7. Process output → update decode state
        let step_output = sched.process_output(&output_tokens, None);

        // 8. Release buffer
        shared.sync.signal_output_consumed();

        // 9. Send StepOutput to scheduler
        let _ = data.send_step_output(&step_output);

        // 10. Heartbeat
        maybe_heartbeat(&control, &sched, &mut last_heartbeat, heartbeat_interval);

        if shared.sync.is_shutdown() { break; }
    }
}

fn maybe_heartbeat(
    control: &ControlPump,
    sched: &SubScheduler,
    last: &mut Instant,
    interval: Duration,
) {
    if last.elapsed() >= interval {
        let active = sched.active_decodes.iter().filter(|s| !s.finished).count();
        let _ = control.send_heartbeat(active);
        *last = Instant::now();
    }
}
