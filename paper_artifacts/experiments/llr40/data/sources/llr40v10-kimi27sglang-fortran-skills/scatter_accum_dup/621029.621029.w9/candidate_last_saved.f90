subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) bind(C, name='scatter_accum_dup_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: bins(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  do i = 1_c_int64_t, LEN_1D
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
end subroutine scatter_accum_dup_fp64
