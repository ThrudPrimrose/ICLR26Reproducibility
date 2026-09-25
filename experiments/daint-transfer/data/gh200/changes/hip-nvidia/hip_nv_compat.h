// Force-included (-include) into every HIP submission built with nvcc for the Daint regrade.
// HIP's NVIDIA backend (ROCm/hip + ROCm/hipother rocm-7.2.4) maps the HIP API to CUDA, but a few
// spellings hipcc accepts on AMD have no CUDA counterpart. Each shim below gives the HIP semantics:
//   - __shfl*/without _sync: HIP shuffles act on the whole wavefront, i.e. a full-warp mask here;
//   - atomicCAS on double/float: HIP overloads, implemented on the same-width integer CAS.
// A submission that relies on a 64-lane wavefront still sees warpSize == 32 and is graded as it is.
#pragma once
#if defined(__CUDACC__)
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template <class T> __device__ __forceinline__ T __shfl(T v, int src, int width = warpSize) {
  return __shfl_sync(0xffffffffu, v, src, width);
}
template <class T> __device__ __forceinline__ T __shfl_up(T v, unsigned int d, int width = warpSize) {
  return __shfl_up_sync(0xffffffffu, v, d, width);
}
template <class T> __device__ __forceinline__ T __shfl_down(T v, unsigned int d, int width = warpSize) {
  return __shfl_down_sync(0xffffffffu, v, d, width);
}
template <class T> __device__ __forceinline__ T __shfl_xor(T v, int m, int width = warpSize) {
  return __shfl_xor_sync(0xffffffffu, v, m, width);
}
#endif

__device__ __forceinline__ double atomicCAS(double* address, double compare, double val) {
  return __longlong_as_double(atomicCAS(reinterpret_cast<unsigned long long*>(address),
                                        static_cast<unsigned long long>(__double_as_longlong(compare)),
                                        static_cast<unsigned long long>(__double_as_longlong(val))));
}
__device__ __forceinline__ float atomicCAS(float* address, float compare, float val) {
  return __int_as_float(atomicCAS(reinterpret_cast<int*>(address), __float_as_int(compare), __float_as_int(val)));
}
#endif
