subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(inout) :: workspace(workspace_size)

  a = a * b * c

end subroutine tsvc_2_vtvtv_fp64
