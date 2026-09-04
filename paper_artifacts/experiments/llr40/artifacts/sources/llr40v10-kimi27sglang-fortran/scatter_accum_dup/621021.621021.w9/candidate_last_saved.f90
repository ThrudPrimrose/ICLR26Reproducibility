subroutine scatter_accum_dup_fp64(bins, ip, src, n, workspace, workspace_bytes) bind(c)
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: bins(n)
  integer(c_int32_t), intent(in) :: ip(n)
  real(c_double), intent(in) :: src(n)
  integer(c_int64_t), value :: n
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i
  integer(c_int32_t) :: idx

  !$omp parallel do
  do i = 1, n
    idx = ip(i) + 1
    !$omp atomic update
    bins(idx) = bins(idx) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
