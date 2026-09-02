subroutine s316(a, result, len_1d, workspace, workspace_size) bind(C, name="tsvc_2_s316_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)
  integer(c_int8_t), intent(inout) :: workspace(*)

  result(1) = minval(a)
end subroutine s316
