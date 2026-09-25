hip-nvidia/include = ROCm/hip rocm-7.2.4 include/hip + ROCm/hipother rocm-7.2.4 hipnv/include/hip/nvidia_detail,
plus the three files here: hip_version.h (stands in for the generated header),
amd_hip_runtime_pt_api.h (empty stub: included unconditionally, AMD-only content),
hip_nv_compat.h (used only by the portability probe, never for grading).
