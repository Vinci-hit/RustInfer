//! Serialization codec abstraction.

use serde::{de::DeserializeOwned, Serialize};

use crate::error::{SchedulerError, TransportError};

/// Serialization/deserialization abstraction.
pub trait Codec: Send + Sync + 'static {
    fn encode<T: Serialize>(&self, value: &T) -> crate::error::Result<Vec<u8>>;
    fn decode<T: DeserializeOwned>(&self, data: &[u8]) -> crate::error::Result<T>;
    fn name(&self) -> &'static str;
}

/// MessagePack codec (current default — compatible with existing wire format).
pub struct MsgPackCodec;

impl Codec for MsgPackCodec {
    fn encode<T: Serialize>(&self, value: &T) -> crate::error::Result<Vec<u8>> {
        rmp_serde::to_vec(value).map_err(|e| {
            SchedulerError::Transport(TransportError::Serialization(format!(
                "msgpack encode: {}",
                e
            )))
        })
    }

    fn decode<T: DeserializeOwned>(&self, data: &[u8]) -> crate::error::Result<T> {
        rmp_serde::from_slice(data).map_err(|e| {
            SchedulerError::Transport(TransportError::Serialization(format!(
                "msgpack decode: {}",
                e
            )))
        })
    }

    fn name(&self) -> &'static str {
        "msgpack"
    }
}
