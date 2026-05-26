//! Runtime domain entities — KvCache, Sequence state machine.

use super::types::Dtype;
use super::tensor::Tensor;
use super::ports::Device;

/// KV Cache — domain entity with business rules (grow strategy, layer structure).
/// Storage is delegated to Infrastructure (via Tensor which uses infra Buffer).
pub struct KvCache<T: Dtype, D: Device> {
    pub k_layers: Vec<Tensor<T, D>>,
    pub v_layers: Vec<Tensor<T, D>>,
    pub capacity: usize,
    pub max_capacity: usize,
    pub kv_dim: usize,
}

impl<T: Dtype, D: Device> KvCache<T, D> {
    #[inline] pub fn capacity(&self) -> usize { self.capacity }
    #[inline] pub fn num_layers(&self) -> usize { self.k_layers.len() }
    #[inline] pub fn kv_dim(&self) -> usize { self.kv_dim }
    pub fn k_caches_mut(&mut self) -> &mut [Tensor<T, D>] { &mut self.k_layers }
    pub fn v_caches_mut(&mut self) -> &mut [Tensor<T, D>] { &mut self.v_layers }

    /// Domain rule: should we grow?
    pub fn needs_growth(&self, required: usize) -> bool { required > self.capacity }
    /// Domain rule: what's the next capacity?
    pub fn next_capacity(&self, required: usize) -> Option<usize> {
        if required <= self.capacity { return None; }
        let mut new_cap = self.capacity;
        while new_cap < required && new_cap < self.max_capacity {
            new_cap = (new_cap * 2).min(self.max_capacity);
        }
        if new_cap >= required { Some(new_cap) } else { None }
    }
}
