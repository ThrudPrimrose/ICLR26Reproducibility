subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: b1

  ! Preserve initial value of b(1) (corresponds to b[0] in C)
  b1 = b(1)

  ! Serial prefix sum for b(i) = b(i-1) + c(i)*(d(i)+e(i)) for i >= 2
  do i = 2, LEN_1D
    b(i) = b(i-1) + c(i) * (d(i) + e(i))
  end do

  ! Compute a(i) = b(i-1) + c(i) * d(i) for i >= 2
  !$omp parallel do simd
  do i = 2, LEN_1D
    a(i) = b(i-1) + c(i) * d(i)
  end do

end subroutine tsvc_2_s323_fp64
