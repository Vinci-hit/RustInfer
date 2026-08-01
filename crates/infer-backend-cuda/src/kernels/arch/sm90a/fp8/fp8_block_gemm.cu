#include <atomic>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

namespace rustinfer_fp8_sm90 {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

// The checkpoint stores weight as row-major [N,K]. CUTLASS's logical B is
// [K,N], so ColumnMajor gives the same byte address n*K+k without a transpose.
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 8;
constexpr int AlignmentD = 8;

using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;

// K-major is intentional. It makes the physical compact layouts exactly
//   SFA [M,K/128] and SFB [N/128,K/128]
// in row-major order, matching both the dynamic activation quantizer and the
// checkpoint's weight_scale_inv tensor.
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    1, 128, 128, GMMA::Major::K, GMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        cute::tuple<LayoutA, LayoutSFA>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutB, LayoutSFB>,
        AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

constexpr int kMaxCachedDevices = 64;
std::atomic<int> sm_count_cache[kMaxCachedDevices]{};
std::atomic<int> max_cluster_cache[kMaxCachedDevices]{};

}  // namespace rustinfer_fp8_sm90

extern "C" int fp8_block_accelerated_init(int device_id) {
  using namespace rustinfer_fp8_sm90;
  if (device_id < 0 || device_id >= kMaxCachedDevices) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }

  if constexpr (GemmKernel::SharedStorageSize >= (48 << 10)) {
    cudaError_t status = cudaFuncSetAttribute(
        cutlass::device_kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GemmKernel::SharedStorageSize);
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }

  auto hw =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);
  sm_count_cache[device_id].store(hw.sm_count, std::memory_order_release);
  max_cluster_cache[device_id].store(
      hw.max_active_clusters, std::memory_order_release);
  return 0;
}

extern "C" int fp8_block_accelerated_bf16(
    const void* activation_fp8,
    const void* weight_fp8,
    const float* activation_scale_inv,
    const float* weight_scale_inv,
    void* output_bf16,
    int M,
    int N,
    int K,
    int device_id,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  using namespace rustinfer_fp8_sm90;
  if (device_id < 0 || device_id >= kMaxCachedDevices) {
    return -1001;
  }

  const auto problem_shape = cute::make_shape(M, N, K, 1);
  const StrideA stride_a{int64_t(K), cute::_1{}, int64_t(M) * K};
  const StrideB stride_b{int64_t(K), cute::_1{}, int64_t(N) * K};
  const StrideC stride_c{int64_t(N), cute::_1{}, int64_t(M) * N};
  const StrideD stride_d{int64_t(N), cute::_1{}, int64_t(M) * N};
  const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

  cutlass::KernelHardwareInfo hw_info{
      device_id,
      sm_count_cache[device_id].load(std::memory_order_acquire),
      max_cluster_cache[device_id].load(std::memory_order_acquire)};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {static_cast<const ElementA*>(activation_fp8),
       stride_a,
       static_cast<const ElementB*>(weight_fp8),
       stride_b,
       activation_scale_inv,
       layout_sfa,
       weight_scale_inv,
       layout_sfb},
      {{1.0f, 0.0f},
       static_cast<const ElementC*>(output_bf16),
       stride_c,
       static_cast<ElementD*>(output_bf16),
       stride_d},
      hw_info};

  if (Gemm::can_implement(arguments) != cutlass::Status::kSuccess) {
    return -1002;
  }
  const size_t required = Gemm::get_workspace_size(arguments);
  if (required > workspace_size || (required != 0 && workspace == nullptr)) {
    return -1003;
  }

  // Do not call Gemm::initialize here: it performs cudaFuncSetAttribute, which
  // is deliberately completed during CudaConfig creation, before graph capture.
  // Workspace initialization is stream-ordered and therefore capture-safe.
  auto status = GemmKernel::initialize_workspace(arguments, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  auto params = GemmKernel::to_underlying_arguments(arguments, workspace);
  status = Gemm::run(params, stream);
  return status == cutlass::Status::kSuccess ? 0 : static_cast<int>(status);
}
