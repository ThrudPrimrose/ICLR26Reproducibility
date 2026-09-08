subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: bins(*)
  real(c_double), intent(in) :: src(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i
  integer(c_int32_t) :: idx

  !$omp parallel do private(idx)
  do i = 1, LEN_1D
    idx = ip(i)
    !$omp atomic
    bins(idx) = bins(idx) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
