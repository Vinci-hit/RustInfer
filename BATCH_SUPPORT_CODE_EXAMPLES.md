# FlashAttnGQA Batch Support: Code Examples

## Example 1: Current Single-Sequence Forward

### Rust Call
```rust
let mut output = Tensor::zeros(&[q_seq_len, q_hidden_dim], device, DataType::F32)?;

self.op.forward(
    &input_q,           // [100, 512] - single sequence
    &k_cache,           // [4096, 256] - full K cache
    &v_cache,           // [4096, 256] - full V cache
    &kv_len_scalar,     // [1] - scalar = 512
    &mut output,        // [100, 512]
    Some(&cuda_config),
)?;
```

### What Happens Inside
```cuda
// CUDA kernel call (simplified)
dim3 grid(8);                    // 8 query heads
dim3 block(64);                  // 64 threads for head_dim

// Kernel execution:
// Block 0 processes head 0 over all Q tokens
// Block 1 processes head 1 over all Q tokens
// ...

// For each token in Q:
//   For each token in KV cache:
//     Compute attention
```

### Key Limitation
```rust
// THIS STRUCTURE DOESN'T SUPPORT BATCHING
// You'd need to call forward() multiple times:

for (i, seq_info) in sequences.iter().enumerate() {
    self.op.forward(
        &seq_info.q,
        &seq_info.k_cache,
        &seq_info.v_cache,
        &seq_info.kv_len,
        &mut output[i],  // ← Separate output slice
        None,
    )?;
}
// ↑ N separate kernel launches, reduced GPU utilization
```

---

## Example 2: Proposed Batched Forward

### Rust Call
```rust
let batch_size = 3;

// Prepare packed sequences (concatenated)
let total_q_tokens = 100 + 50 + 75;     // 225 total
let total_kv_tokens = 512 + 400 + 600;  // 1512 total

let mut packed_q = Tensor::zeros(&[total_q_tokens, q_hidden_dim], device, DataType::F32)?;
let mut packed_output = Tensor::zeros(&[total_q_tokens, q_hidden_dim], device, DataType::F32)?;

// Copy sequences into packed buffers (in order)
// packed_q[0:100] = seq_0.q
// packed_q[100:150] = seq_1.q
// packed_q[150:225] = seq_2.q

// Create cumulative length arrays (on device)
let cu_seqlens_q = Tensor::from_slice(
    &[0, 100, 150, 225],  // Cumulative indices
    &[4],  // batch_size + 1
    device,
    DataType::I32,
)?;

let cu_seqlens_k = Tensor::from_slice(
    &[0, 512, 912, 1512],  // Cumulative KV indices
    &[4],
    device,
    DataType::I32,
)?;

// Single forward call processes all batches
self.op.forward_batched(
    &packed_q,          // [225, 512]
    &packed_k_cache,    // [1512, 256]
    &packed_v_cache,    // [1512, 256]
    &cu_seqlens_q,      // [4] - cumulative Q offsets
    &cu_seqlens_k,      // [4] - cumulative KV offsets
    &mut packed_output, // [225, 512]
    Some(&cuda_config),
)?;

// Result: packed_output[0:100] = seq_0 output, etc.
```

### What Happens Inside
```cuda
// CUDA kernel call (simplified)
dim3 grid(8 * 3);                  // 8 heads × 3 batches = 24 blocks
dim3 block(64);                    // 64 threads for head_dim

// Block indexing:
// Block 0: batch=0, head=0
// Block 1: batch=1, head=0
// Block 2: batch=2, head=0
// Block 3: batch=0, head=1
// ...

// Per-block logic:
int batch_id = blockIdx.x % batch_size;         // 0, 1, or 2
int q_head_idx = blockIdx.x / batch_size;       // 0-7

// Read cumulative offsets
int q_start = cu_seqlens_q[batch_id];           // 0, 100, or 150
int q_end = cu_seqlens_q[batch_id + 1];         // 100, 150, or 225
int kv_start = cu_seqlens_k[batch_id];          // 0, 512, or 912
int kv_end = cu_seqlens_k[batch_id + 1];        // 512, 912, or 1512

// Process only this batch's tokens
for (int q_idx = q_start; q_idx < q_end; ++q_idx) {
    for (int kv_idx = kv_start; kv_idx < kv_end; ++kv_idx) {
        // Compute attention: Q[q_idx] · K[kv_idx]
    }
}
```

### Advantages
- **Single kernel launch** for all batches
- **Better GPU utilization** - all heads working in parallel
- **Scales gracefully** to larger batches

---

## Example 3: Memory Layout (Concrete Numbers)

### Single Sequence Layout (Current)

```
Parameters:
  num_q_heads = 8
  num_kv_heads = 8
  head_dim = 64
  Q_HiddenDim = 8 * 64 = 512
  KV_HiddenDim = 8 * 64 = 512

INPUT TENSOR Q: [100, 512]
Byte layout (flattened):
  Q_buffer[i * 512 + j]  ← 512-byte rows

  Q_buffer[0..511]       = Q token 0
    Q_buffer[0..63]      = Q token 0, head 0
    Q_buffer[64..127]    = Q token 0, head 1
    ...
    Q_buffer[448..511]   = Q token 0, head 7
  
  Q_buffer[512..1023]    = Q token 1
  ...
  Q_buffer[50688..51199] = Q token 99

TOTAL BYTES = 100 * 512 = 51,200

K_CACHE: [4096, 512]
  Only first 512 rows are valid (indicated by kv_len_ptr = 512)
  TOTAL ALLOCATED = 4096 * 512 = 2,097,152 bytes
```

### Batched Layout (Proposed)

```
3 sequences:
  Seq 0: Q_len=100, KV_len=512
  Seq 1: Q_len=50, KV_len=400
  Seq 2: Q_len=75, KV_len=600

PACKED INPUT TENSOR Q: [225, 512]
  Q_buffer[0..51199]     = Seq 0 (100 tokens)
  Q_buffer[51200..76799] = Seq 1 (50 tokens)
  Q_buffer[76800..115199]= Seq 2 (75 tokens)

CU_SEQLENS_Q: [4] = [0, 100, 150, 225]
  cu_seqlens_q[0] = 0     ← Seq 0 starts at token 0
  cu_seqlens_q[1] = 100   ← Seq 1 starts at token 100
  cu_seqlens_q[2] = 150   ← Seq 2 starts at token 150
  cu_seqlens_q[3] = 225   ← Total tokens

PACKED K_CACHE: [1512, 512]
  K_buffer[0..262143]        = Seq 0 KV (512 tokens)
  K_buffer[262144..466943]   = Seq 1 KV (400 tokens)
  K_buffer[466944..779263]   = Seq 2 KV (600 tokens)

CU_SEQLENS_K: [4] = [0, 512, 912, 1512]
  cu_seqlens_k[0] = 0     ← Seq 0 KV starts at 0
  cu_seqlens_k[1] = 512   ← Seq 1 KV starts at 512
  cu_seqlens_k[2] = 912   ← Seq 2 KV starts at 912
  cu_seqlens_k[3] = 1512  ← Total KV tokens

TOTAL BYTES PACKED = 225 * 512 + 1512 * 512 = 115,200 + 774,144 = 889,344 bytes
```

### Address Calculation Examples

#### Single Sequence (Current)
```
Access Q[token_idx=5][head_idx=2][elem_idx=10]:
  Flat index = 5 * 512 + 2 * 64 + 10 = 2560 + 128 + 10 = 2698
  Byte address = Q_buffer + 2698 * 4 = Q_buffer + 10,792 bytes

Access K[token_idx=123][head_idx=3][elem_idx=50]:
  (assuming first 512 rows are valid)
  Flat index = 123 * 512 + 3 * 64 + 50 = 63,009
  Byte address = K_buffer + 63,009 * 4 = K_buffer + 252,036 bytes
```

#### Batched (Proposed)
```
Access Q from batch 1, token 10, head 2, elem 10:
  batch_id = 1
  token_in_batch = 10
  q_global_token = cu_seqlens_q[batch_id] + token_in_batch
                 = 100 + 10 = 110
  Flat index = 110 * 512 + 2 * 64 + 10 = 56,330 + 128 + 10 = 56,468
  Byte address = Q_buffer + 56,468 * 4 = Q_buffer + 225,872 bytes

Access K from batch 1, token 50, head 3, elem 20:
  batch_id = 1
  token_in_batch = 50
  kv_global_token = cu_seqlens_k[batch_id] + token_in_batch
                  = 512 + 50 = 562
  Flat index = 562 * 512 + 3 * 64 + 20 = 287,744 + 192 + 20 = 287,956
  Byte address = K_buffer + 287,956 * 4 = K_buffer + 1,151,824 bytes
```

---

## Example 4: CUDA Kernel Code Snippets

### Current Decode Kernel (Simplified)

```cuda
__global__ void decode_kernel(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* Output,
    int* kv_len_ptr,
    int head_dim,
    int num_kv_heads,
    int group_size
) {
    // One block per Q head (blockIdx.x goes 0..num_q_heads-1)
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / group_size;
    int tid = threadIdx.x;
    
    // Assumes Q_SeqLen = 1 (decode mode)
    float q_val = Q[q_head_idx * head_dim + tid];
    
    // Read KV cache length
    int kv_len = *kv_len_ptr + 1;  // +1 includes current token
    
    float m = -INFINITY;
    float d = 0.0f;
    float out_val = 0.0f;
    
    // Loop over all KV tokens for this single query token
    for (int kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
        int k_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
        float k_val = K_cache[k_offset];
        
        // Compute attention score (Q · K^T)
        float score = q_val * k_val;
        score = __shfl_xor_sync(0xffffffff, score, 32);  // Warp reduce
        
        // Online softmax
        float m_new = max(m, score);
        float scale_old = __expf(m - m_new);
        float scale_new = __expf(score - m_new);
        d = d * scale_old + scale_new;
        
        // Accumulate output
        int v_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
        out_val = out_val * scale_old + V_cache[v_offset] * scale_new;
        
        m = m_new;
    }
    
    // Normalize
    out_val /= d;
    Output[q_head_idx * head_dim + tid] = out_val;
}
```

### Proposed Batched Decode Kernel (Simplified)

```cuda
__global__ void decode_kernel_batched(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* Output,
    const int* cu_seqlens_q,  // [batch_size + 1]
    const int* cu_seqlens_k,  // [batch_size + 1]
    int head_dim,
    int num_q_heads,
    int num_kv_heads,
    int group_size,
    int batch_size
) {
    // Grid is now [num_q_heads * batch_size]
    int global_block = blockIdx.x;
    int batch_id = global_block % batch_size;
    int q_head_idx = global_block / batch_size;
    int kv_head_idx = q_head_idx / group_size;
    int tid = threadIdx.x;
    
    // Get sequence offsets from cumulative arrays
    int q_start = cu_seqlens_q[batch_id];
    int q_end = cu_seqlens_q[batch_id + 1];
    int kv_start = cu_seqlens_k[batch_id];
    int kv_end = cu_seqlens_k[batch_id + 1];
    
    // This batch's sequence lengths
    int q_len = q_end - q_start;           // Should be 1 for decode
    int kv_len = kv_end - kv_start;
    
    if (tid >= head_dim) return;
    
    // Read Q value for this batch/head
    // Q is packed: Q_buffer[(q_start + 0) * num_q_heads * head_dim + q_head_idx * head_dim + tid]
    int q_offset = q_start * num_q_heads * head_dim + q_head_idx * head_dim + tid;
    float q_val = Q[q_offset];
    
    float m = -INFINITY;
    float d = 0.0f;
    float out_val = 0.0f;
    
    // Loop over this batch's KV tokens
    for (int kv_idx = kv_start; kv_idx < kv_end; ++kv_idx) {
        int k_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
        float k_val = K_cache[k_offset];
        
        // Compute attention score
        float score = q_val * k_val;
        score = __shfl_xor_sync(0xffffffff, score, 32);
        
        // Online softmax
        float m_new = max(m, score);
        float scale_old = __expf(m - m_new);
        float scale_new = __expf(score - m_new);
        d = d * scale_old + scale_new;
        
        // Accumulate output
        int v_offset = kv_idx * num_kv_heads * head_dim + kv_head_idx * head_dim + tid;
        out_val = out_val * scale_old + V_cache[v_offset] * scale_new;
        
        m = m_new;
    }
    
    // Normalize
    out_val /= d;
    
    // Write output at correct position
    int out_offset = q_start * num_q_heads * head_dim + q_head_idx * head_dim + tid;
    Output[out_offset] = out_val;
}
```

### Key Differences
```cuda
// OLD: Single batch
int q_head_idx = blockIdx.x;                    // Direct index

// NEW: Multiple batches
int global_block = blockIdx.x;
int batch_id = global_block % batch_size;
int q_head_idx = global_block / batch_size;     // Derived index

// OLD: Fixed KV length
int kv_len = *kv_len_ptr + 1;
for (int kv_idx = 0; kv_idx < kv_len; ++kv_idx)
    int k_offset = kv_idx * ...;

// NEW: Batch-specific KV length and range
int kv_start = cu_seqlens_k[batch_id];
int kv_end = cu_seqlens_k[batch_id + 1];
for (int kv_idx = kv_start; kv_idx < kv_end; ++kv_idx)
    int k_offset = kv_idx * ...;

// OLD: Fixed Q/K offset calculation
Q[q_head_idx * head_dim + tid]

// NEW: Batch-aware offset calculation
Q[(q_start + 0) * num_q_heads * head_dim + q_head_idx * head_dim + tid]
```

---

## Example 5: Rust FFI Changes

### Current FFI Declaration

```rust
unsafe extern "C" {
    pub fn flash_attn_gqa_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        q_seq_len: i32,
        kv_seq_len: *const i32,        // Single pointer
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}
```

### Proposed FFI Addition

```rust
unsafe extern "C" {
    pub fn flash_attn_gqa_varlen_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        cu_seqlens_q: *const i32,      // NEW: Array
        cu_seqlens_k: *const i32,      // NEW: Array
        max_seqlen_q: i32,             // NEW: Max query length
        max_seqlen_k: i32,             // NEW: Max KV length
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        is_causal: i32,
        batch_size: i32,               // NEW: Batch dimension
        stream: cuda::ffi::cudaStream_t,
    );
}
```

### Rust Wrapper Implementation

```rust
pub unsafe fn flash_attn_gqa_varlen(
    input_q: &Tensor,
    input_k_cache: &Tensor,
    input_v_cache: &Tensor,
    output_o: &mut Tensor,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    batch_size: usize,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_causal: bool,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // Validate cu_seqlens
    let cu_q_host = cu_seqlens_q.as_i32()?.as_slice()?;
    let cu_k_host = cu_seqlens_k.as_i32()?.as_slice()?;
    
    if cu_q_host.len() != batch_size + 1 {
        return Err(Error::InvalidArgument(format!(
            "cu_seqlens_q must have length batch_size+1 ({} + 1), got {}",
            batch_size, cu_q_host.len()
        )).into());
    }
    
    // Verify monotonicity and bounds
    if cu_q_host[0] != 0 || cu_k_host[0] != 0 {
        return Err(Error::InvalidArgument("cu_seqlens must start with 0".into()).into());
    }
    
    if cu_q_host[batch_size] != input_q.shape()[0] as i32 {
        return Err(Error::InvalidArgument(format!(
            "cu_seqlens_q final value must match total Q tokens: {} vs {}",
            cu_q_host[batch_size], input_q.shape()[0]
        )).into());
    }
    
    // Get pointers
    let q_ptr = input_q.as_f32()?.buffer().as_ptr() as *const f32;
    let k_ptr = input_k_cache.as_f32()?.buffer().as_ptr() as *const f32;
    let v_ptr = input_v_cache.as_f32()?.buffer().as_ptr() as *const f32;
    let o_ptr = output_o.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
    
    let cu_q_ptr = cu_seqlens_q.as_i32()?.buffer().as_ptr() as *const i32;
    let cu_k_ptr = cu_seqlens_k.as_i32()?.buffer().as_ptr() as *const i32;
    
    let stream = CudaConfig::resolve_stream(cuda_config);
    let is_causal_i32 = if is_causal { 1 } else { 0 };
    
    // Launch kernel
    flash_attn_gqa_varlen_cu(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        cu_q_ptr,
        cu_k_ptr,
        max_seqlen_q as i32,
        max_seqlen_k as i32,
        num_q_heads as i32,
        num_kv_heads as i32,
        head_dim as i32,
        is_causal_i32,
        batch_size as i32,
        stream,
    );
    
    Ok(())
}
```

---

## Example 6: Rust Op Layer Integration

### New Method on FlashAttnGQA

```rust
impl FlashAttnGQA {
    /// Forward pass with batched variable-length sequences (new)
    pub fn forward_batched(
        &self,
        input_q: &Tensor,                   // [TotalQ, Q_HiddenDim]
        input_k_cache: &Tensor,             // [TotalKV, KV_HiddenDim]
        input_v_cache: &Tensor,             // [TotalKV, KV_HiddenDim]
        cu_seqlens_q: &Tensor,              // [Batch+1]
        cu_seqlens_k: &Tensor,              // [Batch+1]
        output_o: &mut Tensor,              // [TotalQ, Q_HiddenDim]
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let device = input_q.device();
        let batch_size = cu_seqlens_q.shape()[0] - 1;
        
        // Get total lengths from cumulative arrays
        let cu_q_slice = cu_seqlens_q.as_i32()?.as_slice()?;
        let cu_k_slice = cu_seqlens_k.as_i32()?.as_slice()?;
        
        let total_q = cu_q_slice[batch_size] as usize;
        let total_k = cu_k_slice[batch_size] as usize;
        
        // Calculate max sequence lengths
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        for i in 0..batch_size {
            let q_len = (cu_q_slice[i + 1] - cu_q_slice[i]) as usize;
            let k_len = (cu_k_slice[i + 1] - cu_k_slice[i]) as usize;
            max_seqlen_q = max_seqlen_q.max(q_len);
            max_seqlen_k = max_seqlen_k.max(k_len);
        }
        
        // Validation
        if input_q.shape()[0] != total_q {
            return Err(Error::InvalidArgument(format!(
                "input_q shape mismatch: {} vs cu_seqlens_q[-1]={}",
                input_q.shape()[0], total_q
            )).into());
        }
        
        if input_k_cache.shape()[0] != total_k {
            return Err(Error::InvalidArgument(format!(
                "input_k_cache shape mismatch: {} vs cu_seqlens_k[-1]={}",
                input_k_cache.shape()[0], total_k
            )).into());
        }
        
        match device {
            DeviceType::Cpu => {
                kernels::cpu::flash_attn_gqa_batched(
                    input_q,
                    input_k_cache,
                    input_v_cache,
                    output_o,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    batch_size,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.causal,
                )?;
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                unsafe {
                    kernels::cuda::flash_attn_gqa_varlen(
                        input_q,
                        input_k_cache,
                        input_v_cache,
                        output_o,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        batch_size,
                        max_seqlen_q,
                        max_seqlen_k,
                        self.num_q_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.causal,
                        cuda_config,
                    )?;
                }
            }
            #[cfg(not(feature = "cuda"))]
            _ => return Err(Error::Unimplemented("Device type not supported.".into())),
        }
        
        Ok(())
    }
}
```

---

## Conclusion

These examples show:

1. **API Interface**: How users would call the new batched function
2. **Memory Layout**: Concrete byte-level organization for packed sequences
3. **CUDA Kernels**: How indexing changes from single to batched
4. **FFI Layer**: New C/CUDA function signatures
5. **Rust Wrapper**: Full integration with validation and dispatch

The key insight is that **all changes are additive** — the new `forward_batched()` method can coexist with the current `forward()`, providing a smooth migration path.

