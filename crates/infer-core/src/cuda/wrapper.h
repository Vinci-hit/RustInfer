#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include "../op/kernels/cuda/total_head.h"

#ifndef cublasCreate_v2
#define cublasCreate_v2 cublasCreate
#define cublasDestroy_v2 cublasDestroy
#endif