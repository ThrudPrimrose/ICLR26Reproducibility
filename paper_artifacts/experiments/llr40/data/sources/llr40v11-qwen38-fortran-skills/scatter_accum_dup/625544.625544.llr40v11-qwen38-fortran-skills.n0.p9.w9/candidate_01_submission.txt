subroutine scatter_accum_dup_fp64(bins, ip, src, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: bins(len_1d)
  integer(c_int32_t), intent(in) :: ip(len_1d)
  real(c_double), intent(in) :: src(len_1d)
  type(c_ptr) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i

  !$omp parallel do
  do i = 1, len_1d
    !$omp atomic
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
end subroutine scatter_accum_dup_fp64
