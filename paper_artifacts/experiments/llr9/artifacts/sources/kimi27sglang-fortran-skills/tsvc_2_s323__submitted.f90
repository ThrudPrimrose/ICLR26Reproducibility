subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: cd, ce

  do i = 2_c_int64_t, LEN_1D
    cd = c(i) * d(i)
    a(i) = b(i - 1_c_int64_t) + cd
    ce = c(i) * e(i)
    b(i) = a(i) + ce
  end do
end subroutine tsvc_2_s323_fp64
