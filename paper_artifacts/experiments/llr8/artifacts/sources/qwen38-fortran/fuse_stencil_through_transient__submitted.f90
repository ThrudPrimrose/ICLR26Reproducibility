subroutine fuse_stencil_through_transient_fp64(a, out, len_1d, workspace, workspace_size) &
    bind(C, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(inout) :: out(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: j

  if (len_1d < 4) return
  do j = 2, len_1d - 2
    out(j) = (a(j-1) + a(j) + a(j+1)) * (a(j) + a(j+1) + a(j+2))
  end do
end subroutine fuse_stencil_through_transient_fp64
