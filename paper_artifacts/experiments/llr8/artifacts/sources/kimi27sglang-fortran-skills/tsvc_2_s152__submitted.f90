subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D)
  real(c_double), intent(in) :: e(LEN_1D)
  integer(c_int64_t) :: i

  do concurrent(i = 1:LEN_1D)
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
end subroutine tsvc_2_s152_fp64
