subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(in)    :: c(*)
  real(c_double), intent(in)    :: d(*)
  real(c_double), intent(in)    :: e(*)
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int64_t) :: i

  do i = 2, LEN_1D
    a(i) = b(i-1) + c(i)*d(i)
    b(i) = a(i) + c(i)*e(i)
  end do
end subroutine tsvc_2_s323_fp64
