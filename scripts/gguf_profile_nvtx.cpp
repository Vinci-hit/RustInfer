// Optional NVTX bridge for the standalone Rust profiling binary.
#include <nvtx3/nvToolsExt.h>
extern "C" void ri_nvtx_push(const char * name) { nvtxRangePushA(name); }
extern "C" void ri_nvtx_pop() { nvtxRangePop(); }
