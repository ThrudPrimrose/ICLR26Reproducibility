subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(c, name="tsvc_2_s323_fp64")
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i
  do i = 2, LEN_1D
    a(i) = b(i - 1) + c(i) * d(i)
    b(i) = a(i) + c(i) * e(i)
  end do
end subroutine tsvc_2_s323_fp64
