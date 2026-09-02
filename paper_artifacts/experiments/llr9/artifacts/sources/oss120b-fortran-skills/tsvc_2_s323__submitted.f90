subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s323_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i

  ! Compute a and b sequentially with correct order
  do i = 2, LEN_1D
    a(i) = b(i-1) + c(i) * d(i)
    b(i) = a(i) + c(i) * e(i)
  end do

end subroutine tsvc_2_s323_fp64
