subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(c, name='tsvc_2_s1244_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(inout) :: d(LEN_1D)
  integer(c_int64_t) :: i

  do i = 1, LEN_1D - 1
    a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    d(i) = a(i) + a(i + 1)
  end do
end subroutine tsvc_2_s1244_fp64
