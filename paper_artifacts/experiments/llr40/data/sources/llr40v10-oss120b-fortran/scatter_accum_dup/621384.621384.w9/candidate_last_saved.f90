module scatter_accum_dup_mod
  use iso_c_binding
   use omp_lib
  implicit none
contains
  subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, ws, ws_bytes) bind(C, name="scatter_accum_dup_fp64")
    real(C_DOUBLE), intent(inout) :: bins(*)
    real(C_DOUBLE), intent(in) :: src(*)
    integer(C_INT32_T), intent(in) :: ip(*)
    integer(C_INT64_T), value :: LEN_1D
    type(C_PTR), intent(in) :: ws
    integer(C_INT64_T), value :: ws_bytes
    integer(C_INT64_T) :: i
    integer(C_INT) :: idx
    real(C_DOUBLE), allocatable :: local_sum(:)
    ! Allocate a reduction array for per-thread private copies
    allocate(local_sum(LEN_1D))
    !$omp parallel do private(i, idx) reduction(+:local_sum) shared(src, ip, LEN_1D)
    do i = 0_C_INT64_T, LEN_1D-1_C_INT64_T
        idx = ip(i+1) + 1
        local_sum(idx) = local_sum(idx) + src(i+1)
    end do
    !$omp end parallel do
    ! Combine contributions into bins
    bins(1:LEN_1D) = bins(1:LEN_1D) + local_sum(1:LEN_1D)
    deallocate(local_sum)
  end subroutine scatter_accum_dup_fp64
end module scatter_accum_dup_mod
