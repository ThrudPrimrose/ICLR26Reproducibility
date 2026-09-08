subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D)
  real(c_double), intent(in) :: e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: term, bi
  real(c_double) :: prefix
  prefix = 0.0d0
  do i = 2, LEN_1D
    term = c(i) * (d(i) + e(i))
    prefix = prefix + term
    bi = b(1) + prefix
    b(i) = bi
    a(i) = b(i) - c(i) * e(i)
  end do
end subroutine tsvc_2_s323_fp64
