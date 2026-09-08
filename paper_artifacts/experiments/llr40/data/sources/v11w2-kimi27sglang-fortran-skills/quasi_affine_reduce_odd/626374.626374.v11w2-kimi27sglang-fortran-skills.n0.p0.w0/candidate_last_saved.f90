subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: out(1)
  integer(c_int8_t), intent(inout) :: workspace(*)

  out(1) = sum(a(2:LEN_1D:2))
end subroutine quasi_affine_reduce_odd_fp64
