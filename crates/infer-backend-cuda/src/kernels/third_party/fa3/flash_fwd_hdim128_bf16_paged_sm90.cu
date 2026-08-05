// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"
#include "rustinfer_fa3_api.h"

#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, true, false, true>(Flash_fwd_params &params, cudaStream_t stream);

namespace {

template <bool IsCausal, bool Varlen>
cudaError_t configure_rustinfer_fa3_variant(int max_dynamic_smem) {
    return configure_flash_fwd_kernel<
        90, 128, 128, 1,
        cutlass::bfloat16_t, cutlass::bfloat16_t,
        IsCausal, false, false, Varlen, true, false, false, true, false, false>(
            max_dynamic_smem);
}

}  // namespace
#endif

extern "C" int rustinfer_fa3_init_kernel_attributes(int max_dynamic_smem) {
#ifndef FLASHATTENTION_DISABLE_HDIM128
    // CUTLASS declares device_kernel as `__global__ static`, so its function
    // identity is translation-unit local. Keep these attribute calls beside
    // the run_mha_fwd_ instantiation that launches the same kernel symbols.
    // The RustInfer adapter currently selects causal+varlen; initialize every
    // variant emitted by this run_mha_fwd_ instantiation to keep its dispatch
    // contract safe if the adapter broadens later.
    cudaError_t status = configure_rustinfer_fa3_variant<true, true>(max_dynamic_smem);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = configure_rustinfer_fa3_variant<true, false>(max_dynamic_smem);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = configure_rustinfer_fa3_variant<false, true>(max_dynamic_smem);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = configure_rustinfer_fa3_variant<false, false>(max_dynamic_smem);
    return static_cast<int>(status);
#else
    (void)max_dynamic_smem;
    return static_cast<int>(cudaSuccess);
#endif
}
