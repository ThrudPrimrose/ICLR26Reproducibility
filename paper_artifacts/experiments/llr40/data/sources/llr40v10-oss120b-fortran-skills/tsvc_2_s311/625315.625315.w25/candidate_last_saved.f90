subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: sum_out(1)
  sum_out(1) = sum(a)
end subroutine tsvc_2_s311_fp64
