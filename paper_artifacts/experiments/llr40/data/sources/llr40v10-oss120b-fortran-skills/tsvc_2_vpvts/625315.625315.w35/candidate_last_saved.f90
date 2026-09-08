subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: S
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in)    :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: s_val
  s_val = real(S, c_double)
  a(1:LEN_1D) = a(1:LEN_1D) + b(1:LEN_1D) * s_val
end subroutine tsvc_2_vpvts_fp64
