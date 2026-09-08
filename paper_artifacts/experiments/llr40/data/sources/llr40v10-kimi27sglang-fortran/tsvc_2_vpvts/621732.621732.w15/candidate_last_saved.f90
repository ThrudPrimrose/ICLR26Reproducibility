subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D, S
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: scale

  scale = real(S, c_double)
  do i = 1, LEN_1D
     a(i) = a(i) + b(i) * scale
  end do
end subroutine tsvc_2_vpvts_fp64
