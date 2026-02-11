//! Attention Backend — model-level resource shared by all attention layers.
//!
//! Wraps FlashInfer's `BatchDecodeHandler` with a Rust-friendly Plan/Run interface.
//! Follows SGLang/vLLM architecture: one handler per model, Plan once per forward,
//! Run once per layer.

#[cfg(feature = "cuda")]
use crate::op::kernels::cuda::flashinfer::{BatchDecodeHandler, WorkspaceSizes};
#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::base::{DataType, DeviceType};

/// Attention backend — model-level resource, all attention layers share one instance.
pub struct AttentionBackend {
    #[cfg(feature = "cuda")]
    handler: BatchDecodeHandler,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    page_size: u32,
    /// Plan output: cached last_page_len (host side, computed per plan)
    last_page_len_host: Vec<i32>,
    /// Plan output: cached page_indptr (host side, computed per plan)
    page_indptr_host: Vec<i32>,
    /// Device-side last_page_len tensor (fixed address for CudaGraph)
    last_page_len_device: Tensor, // [max_batch_size] I32
    /// Device-side page_indptr tensor (fixed address for CudaGraph)
    page_indptr_device: Tensor, // [max_batch_size + 1] I32
}

// Safety: AttentionBackend is only used from the model's forward path,
// which is single-threaded. The raw pointers inside BatchDecodeHandler
// are CUDA resources that are thread-safe when accessed sequentially.
unsafe impl Sync for AttentionBackend {}

impl AttentionBackend {
    /// Create a new AttentionBackend.
    ///
    /// # Arguments
    /// * `num_qo_heads` - Number of query/output heads
    /// * `num_kv_heads` - Number of key/value heads
    /// * `head_dim` - Dimension per head
    /// * `page_size` - KV cache page (block) size
    /// * `max_batch_size` - Maximum batch size
    /// * `max_seq_len` - Maximum sequence length
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        page_size: u32,
        max_batch_size: u32,
        max_seq_len: u32,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let handler = {
            let sizes = WorkspaceSizes::calculate(
                max_batch_size,
                max_seq_len,
                num_qo_heads,
                head_dim,
            );
            BatchDecodeHandler::with_workspace_sizes(sizes)?
        };

        let last_page_len_host = vec![0i32; max_batch_size as usize];
        let page_indptr_host = vec![0i32; (max_batch_size + 1) as usize];

        // Allocate device tensors with fixed addresses (for CudaGraph compatibility)
        let device = DeviceType::Cuda(0);
        let last_page_len_device = Tensor::new(
            &[max_batch_size as usize],
            DataType::I32,
            device,
        )?;
        let page_indptr_device = Tensor::new(
            &[(max_batch_size + 1) as usize],
            DataType::I32,
            device,
        )?;

        Ok(Self {
            #[cfg(feature = "cuda")]
            handler,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            last_page_len_host,
            page_indptr_host,
            last_page_len_device,
            page_indptr_device,
        })
    }

    /// Plan: called once per forward_paged invocation.
    ///
    /// Receives token-level kv_indptr (host), internally converts to page-level
    /// metadata and calls the FlashInfer handler's plan.
    ///
    /// # Arguments
    /// * `kv_indptr_host` - [batch_size + 1] token-level CSR indptr (host memory)
    /// * `batch_size` - Number of requests in the batch
    /// * `enable_cuda_graph` - Whether to enable CudaGraph compatibility mode
    /// * `stream` - CUDA stream for async operations
    #[cfg(feature = "cuda")]
    pub fn plan(
        &mut self,
        kv_indptr_host: &[i32],
        batch_size: u32,
        enable_cuda_graph: bool,
        stream: cudaStream_t,
    ) -> Result<()> {
        if kv_indptr_host.len() != (batch_size + 1) as usize {
            return Err(Error::InvalidArgument(format!(
                "kv_indptr_host length {} doesn't match batch_size + 1 = {}",
                kv_indptr_host.len(),
                batch_size + 1
            )).into());
        }

        // 1. Compute kv_len[i] = indptr[i+1] - indptr[i]
        // 2. Compute page_indptr: num_pages[i] = ceil(kv_len[i] / page_size)
        // 3. Compute last_page_len[i] = ((kv_len[i] - 1) % page_size) + 1
        let bs = batch_size as usize;
        self.page_indptr_host[0] = 0;
        for i in 0..bs {
            let kv_len = kv_indptr_host[i + 1] - kv_indptr_host[i];
            let num_pages = (kv_len as u32 + self.page_size - 1) / self.page_size;
            self.page_indptr_host[i + 1] = self.page_indptr_host[i] + num_pages as i32;

            if kv_len > 0 {
                self.last_page_len_host[i] = ((kv_len - 1) % self.page_size as i32) + 1;
            } else {
                self.last_page_len_host[i] = 0;
            }
        }

        // 4. Copy last_page_len to device tensor
        {
            let mut lpl_slice = self.last_page_len_device.slice(&[0], &[bs])?;
            let src = Tensor::from_slice(&self.last_page_len_host[..bs], DeviceType::Cpu)?;
            lpl_slice.copy_from(&src)?;
        }

        // 5. Copy page_indptr to device tensor
        {
            let mut indptr_slice = self.page_indptr_device.slice(&[0], &[bs + 1])?;
            let src = Tensor::from_slice(&self.page_indptr_host[..bs + 1], DeviceType::Cpu)?;
            indptr_slice.copy_from(&src)?;
        }

        // 6. Call handler.plan with page-level indptr
        self.handler.plan(
            &self.page_indptr_host[..bs + 1],
            batch_size,
            self.num_qo_heads,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
            enable_cuda_graph,
            stream,
        )?;

        Ok(())
    }

    /// Run decode attention: called once per layer.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch_size, num_qo_heads * head_dim]
    /// * `k_cache` - Key cache [total_slots, kv_heads, head_dim]
    /// * `v_cache` - Value cache [total_slots, kv_heads, head_dim]
    /// * `output` - Output tensor [batch_size, num_qo_heads * head_dim]
    /// * `kv_indices` - Page indices [nnz] (device, from kv_indptr_device token-level)
    /// * `batch_size` - Number of requests
    /// * `stream` - CUDA stream
    ///
    /// # Safety
    /// All tensors must be valid device tensors with correct shapes.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_decode(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        output: &mut Tensor,
        kv_indices: &Tensor,
        batch_size: u32,
        stream: cudaStream_t,
    ) -> Result<()> {
        let bs = batch_size as usize;

        // Get page_indptr device pointer
        let page_indptr_view = self.page_indptr_device.slice(&[0], &[bs + 1])?;
        let page_indptr_ptr = page_indptr_view.as_i32()?.buffer().as_ptr() as *const i32;

        // Get last_page_len device pointer
        let lpl_view = self.last_page_len_device.slice(&[0], &[bs])?;
        let lpl_ptr = lpl_view.as_i32()?.buffer().as_ptr() as *const i32;

        // Get kv_indices device pointer
        let kv_indices_ptr = kv_indices.as_i32()?.buffer().as_ptr() as *const i32;

        // Dispatch based on dtype
        match q.dtype() {
            DataType::BF16 => {
                let q_ptr = q.as_bf16()?.buffer().as_ptr() as *const half::bf16;
                let k_ptr = k_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
                let v_ptr = v_cache.as_bf16()?.buffer().as_ptr() as *const half::bf16;
                let o_ptr = output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

                unsafe {
                    self.handler.run_bf16(
                        q_ptr, k_ptr, v_ptr, o_ptr,
                        kv_indices_ptr, page_indptr_ptr, lpl_ptr,
                        batch_size, stream,
                    )?;
                }
            }
            other => {
                return Err(Error::InvalidArgument(format!(
                    "AttentionBackend only supports BF16, got {:?}",
                    other
                )).into());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_metadata_computation() {
        // page_size = 16
        // Request 0: kv_len = 33 → 3 pages, last_page_len = 1
        // Request 1: kv_len = 16 → 1 page, last_page_len = 16
        // Request 2: kv_len = 1  → 1 page, last_page_len = 1
        let page_size: u32 = 16;
        let batch_size: u32 = 3;

        // Token-level indptr: [0, 33, 49, 50]
        let kv_indptr = [0i32, 33, 49, 50];

        let mut page_indptr = vec![0i32; (batch_size + 1) as usize];
        let mut last_page_len = vec![0i32; batch_size as usize];

        for i in 0..batch_size as usize {
            let kv_len = kv_indptr[i + 1] - kv_indptr[i];
            let num_pages = (kv_len as u32 + page_size - 1) / page_size;
            page_indptr[i + 1] = page_indptr[i] + num_pages as i32;
            if kv_len > 0 {
                last_page_len[i] = ((kv_len - 1) % page_size as i32) + 1;
            }
        }

        assert_eq!(page_indptr, [0, 3, 4, 5]);
        assert_eq!(last_page_len, [1, 16, 1]);
    }

    #[test]
    fn test_page_metadata_empty_request() {
        let page_size: u32 = 16;
        let kv_indptr = [0i32, 0, 16];

        let mut page_indptr = vec![0i32; 3];
        let mut last_page_len = vec![0i32; 2];

        for i in 0..2 {
            let kv_len = kv_indptr[i + 1] - kv_indptr[i];
            let num_pages = (kv_len as u32 + page_size - 1) / page_size;
            page_indptr[i + 1] = page_indptr[i] + num_pages as i32;
            if kv_len > 0 {
                last_page_len[i] = ((kv_len - 1) % page_size as i32) + 1;
            }
        }

        assert_eq!(page_indptr, [0, 0, 1]);
        assert_eq!(last_page_len, [0, 16]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_backend_creation() {
        let backend = AttentionBackend::new(32, 8, 128, 16, 64, 4096);
        assert!(backend.is_ok(), "Backend creation should succeed");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_plan_lifecycle() -> Result<()> {
        let mut backend = AttentionBackend::new(32, 8, 128, 16, 4, 4096)?;

        // Plan with batch_size=2, each request has 16 tokens
        let kv_indptr = [0i32, 16, 32];
        let stream: cudaStream_t = std::ptr::null_mut();
        backend.plan(&kv_indptr, 2, false, stream)?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_plan_invalid_indptr() {
        let mut backend = AttentionBackend::new(32, 8, 128, 16, 4, 4096).unwrap();
        let kv_indptr = [0i32, 16]; // length 2 but batch_size=2 expects 3
        let stream: cudaStream_t = std::ptr::null_mut();

        let result = backend.plan(&kv_indptr, 2, false, stream);
        assert!(result.is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_run_decode_stub() -> Result<()> {
        let batch_size: u32 = 2;
        let num_qo_heads: u32 = 4;
        let num_kv_heads: u32 = 4;
        let head_dim: u32 = 128;
        let page_size: u32 = 16;
        let dim = (num_qo_heads * head_dim) as usize;
        let total_slots = 64usize; // enough for test

        let mut backend = AttentionBackend::new(
            num_qo_heads, num_kv_heads, head_dim, page_size, batch_size, 4096,
        )?;

        // Create tensors on device
        let device = DeviceType::Cuda(0);
        let q = Tensor::new(&[batch_size as usize, dim], DataType::BF16, device)?;
        let k_cache = Tensor::new(&[total_slots, num_kv_heads as usize, head_dim as usize], DataType::BF16, device)?;
        let v_cache = Tensor::new(&[total_slots, num_kv_heads as usize, head_dim as usize], DataType::BF16, device)?;
        let mut output = Tensor::new(&[batch_size as usize, dim], DataType::BF16, device)?;

        // kv_indices: 2 pages (one per request)
        let kv_indices_host = Tensor::from_slice(&[0i32, 1], DeviceType::Cpu)?;
        let mut kv_indices = Tensor::new(&[2], DataType::I32, device)?;
        kv_indices.copy_from(&kv_indices_host)?;

        // Plan: each request has page_size tokens (1 page each)
        let kv_indptr = [0i32, page_size as i32, 2 * page_size as i32];
        let stream: cudaStream_t = std::ptr::null_mut();
        backend.plan(&kv_indptr, batch_size, false, stream)?;

        // Run decode (stub: copies Q → O)
        unsafe {
            backend.run_decode(
                &q, &k_cache, &v_cache, &mut output,
                &kv_indices, batch_size, stream,
            )?;
        }

        Ok(())
    }
}
