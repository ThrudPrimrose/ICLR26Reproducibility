subroutine scatter_accum_dup_fp64(bins, src, ip, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: bins(*)
  real(c_double), intent(in)    :: src(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int64_t), value    :: LEN_1D
  integer(c_int64_t) :: i, n, idx64
  ! integer idx removed
  n = LEN_1D
  !$omp parallel do default(none) shared(bins, src, ip, n) private(i, idx64) schedule(static)
  do i = 1_c_int64_t, n
    idx64 = int(ip(i), kind=c_int64_t) + 1_c_int64_t
    !idx conversion removed   ! convert to default integer for indexing
    !$omp atomic
    bins(idx64) = bins(idx64) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
