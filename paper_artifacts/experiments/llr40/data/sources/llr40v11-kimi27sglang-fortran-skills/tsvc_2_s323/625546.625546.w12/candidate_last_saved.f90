subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: ai, bi

  if (LEN_1D < 2) return

  bi = b(1)
  do i = 2, LEN_1D
    ai = bi + c(i) * d(i)
    bi = ai + c(i) * e(i)
    a(i) = ai
    b(i) = bi
  end do
end subroutine tsvc_2_s323_fp64
