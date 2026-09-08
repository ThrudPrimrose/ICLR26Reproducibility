subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_bytes) bind(c)
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_int8_t
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(inout) :: bins(0:LEN_1D-1)
  integer(c_int32_t), intent(in) :: ip(0:LEN_1D-1)
  real(c_double), intent(in) :: src(0:LEN_1D-1)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i
  do i = 0, LEN_1D-1
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
end subroutine scatter_accum_dup_fp64
