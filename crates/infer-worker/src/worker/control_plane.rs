//! Control Plane Client
//!
//! Handles ZeroMQ communication between Worker and Scheduler.
//! Encapsulates connection, framing, serialization, and handshake.
//!
//! ```text
//! ┌─────────────────────────────┐
//! │        Scheduler            │
//! │      (RouterSocket)         │
//! └─────────────┬───────────────┘
//!               │ ZeroMQ (IPC/TCP)
//!               ▼
//! ┌─────────────────────────────┐
//! │    ControlPlaneClient       │
//! │      (DealerSocket)         │
//! │                             │
//! │  send_response(resp) ──────►│  [empty_frame, bincode(resp)]
//! │  recv_command() ◄───────────│  [empty_frame, bincode(cmd)]
//! │  handshake()                │  Register → RegisterAck
//! └─────────────────────────────┘
//! ```

use anyhow::Result;
use zeromq::{Socket, SocketRecv, SocketSend, ZmqMessage};

use infer_protocol::{
    WorkerCommand, WorkerResponse,
    WorkerRegistration, WorkerRegistrationAck,
};

/// Control plane client for Scheduler communication.
///
/// Wraps a ZeroMQ DealerSocket with bincode serialization
/// and the RustInfer framing convention.
pub struct ControlPlaneClient {
    socket: zeromq::DealerSocket,
    rank: usize,
    world_size: usize,
}

impl ControlPlaneClient {
    /// Connect to the Scheduler.
    ///
    /// Creates a DealerSocket and connects to the given endpoint.
    pub async fn connect(
        rank: usize,
        world_size: usize,
        scheduler_url: &str,
    ) -> Result<Self> {
        let mut socket = zeromq::DealerSocket::new();
        println!("[Worker-{}] Connecting to scheduler at {}", rank, scheduler_url);
        socket.connect(scheduler_url).await?;
        println!("[Worker-{}] Connected successfully", rank);

        Ok(Self {
            socket,
            rank,
            world_size,
        })
    }

    /// Send a WorkerResponse to the Scheduler.
    pub async fn send_response(&mut self, response: WorkerResponse) -> Result<()> {
        let payload = bincode::serialize(&response)?;
        if payload.is_empty() {
            return Err(anyhow::anyhow!("Cannot send empty response payload"));
        }
        self.send_raw(&payload).await
    }

    /// Send a WorkerCommand to the Scheduler (used during handshake).
    pub async fn send_command(&mut self, command: WorkerCommand) -> Result<()> {
        let payload = bincode::serialize(&command)?;
        self.send_raw(&payload).await
    }

    /// Receive the next WorkerCommand from the Scheduler.
    ///
    /// Blocks until a message arrives. Returns `Err` on ZMQ or deserialization failure.
    pub async fn recv_command(&mut self) -> Result<WorkerCommand> {
        let payload = self.recv_raw().await?;
        let command: WorkerCommand = bincode::deserialize(&payload)?;
        Ok(command)
    }

    /// Receive the next WorkerResponse from the Scheduler (used during handshake).
    pub async fn recv_response(&mut self) -> Result<WorkerResponse> {
        let payload = self.recv_raw().await?;
        let response: WorkerResponse = bincode::deserialize(&payload)?;
        Ok(response)
    }

    /// Receive a WorkerCommand with a timeout.
    ///
    /// Returns `Ok(None)` if the timeout expires.
    pub async fn recv_command_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<Option<WorkerCommand>> {
        match tokio::time::timeout(timeout, self.recv_command()).await {
            Ok(Ok(cmd)) => Ok(Some(cmd)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None), // timeout
        }
    }

    /// Perform handshake: send Register, wait for RegisterAck.
    ///
    /// Returns the ack on success, or an error on timeout / rejection.
    pub async fn handshake(
        &mut self,
        worker_id: &str,
        device_type: &str,
        device_id: u32,
    ) -> Result<WorkerRegistrationAck> {
        println!("[Worker-{}] Performing handshake with Scheduler...", self.rank);

        let registration = WorkerRegistration {
            worker_id: worker_id.to_string(),
            rank: self.rank as u32,
            world_size: self.world_size as u32,
            device_type: device_type.to_string(),
            device_id,
            protocol_version: "0.1.0".to_string(),
        };

        // Send Register command
        let register_cmd = WorkerCommand::Register(registration.clone());
        self.send_command(register_cmd).await?;
        println!(
            "[Worker-{}] Sent registration: worker_id={}, rank={}, device={}:{}",
            self.rank, registration.worker_id, registration.rank,
            registration.device_type, registration.device_id,
        );

        // Wait for ACK with timeout
        let timeout = std::time::Duration::from_secs(10);
        let response = match tokio::time::timeout(timeout, self.recv_response()).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(e)) => {
                return Err(anyhow::anyhow!("ZMQ error during handshake: {}", e));
            }
            Err(_) => {
                return Err(anyhow::anyhow!("Handshake timeout after 10 seconds"));
            }
        };

        // Validate response
        match response {
            WorkerResponse::RegisterAck(ack) => {
                if ack.status == "ok" {
                    println!("[Worker-{}] Handshake successful: {}", self.rank, ack.message);
                    Ok(ack)
                } else {
                    Err(anyhow::anyhow!("Handshake rejected: {}", ack.message))
                }
            }
            other => {
                Err(anyhow::anyhow!(
                    "Unexpected response during handshake: {:?}",
                    other
                ))
            }
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    // -----------------------------------------------------------------------
    // Internal: raw ZMQ framing
    // -----------------------------------------------------------------------

    /// Send raw bytes over the DealerSocket.
    ///
    /// DealerSocket framing: `[empty_delimiter_frame, payload_frame]`
    async fn send_raw(&mut self, payload: &[u8]) -> Result<()> {
        let mut msg = ZmqMessage::try_from(Vec::<u8>::new())?; // empty delimiter
        msg.push_back(payload.to_vec().into());
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Receive raw bytes from the DealerSocket.
    ///
    /// Expects `[empty_delimiter_frame, payload_frame]` and returns the payload.
    async fn recv_raw(&mut self) -> Result<Vec<u8>> {
        let msg: ZmqMessage = self.socket.recv().await?;
        let frames = msg.into_vec();

        if frames.len() < 2 {
            // Some messages may come as a single frame (no delimiter)
            if frames.len() == 1 {
                return Ok(frames.into_iter().next().unwrap().to_vec());
            }
            return Err(anyhow::anyhow!(
                "Invalid ZMQ message: expected >=1 frame, got {}",
                frames.len()
            ));
        }

        // frames[0] = empty delimiter, frames[1] = payload
        Ok(frames[1].to_vec())
    }
}
