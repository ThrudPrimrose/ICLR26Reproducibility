subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(c, name='tsvc_2_vpvts_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int64_t), intent(in), value :: S
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: s_d

  s_d = real(S, c_double)
  do i = 1, LEN_1D
    a(i) = a(i) + b(i) * s_d
  end do
end subroutine tsvc_2_vpvts_fp64
