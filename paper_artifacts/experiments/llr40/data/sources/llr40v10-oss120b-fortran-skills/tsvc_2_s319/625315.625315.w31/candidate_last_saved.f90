subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(out) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D)
  real(c_double), intent(in) :: e(LEN_1D)

  integer(c_int64_t) :: i
  real(c_double) :: sum_val

  sum_val = 0.0_c_double

  !$omp parallel do simd reduction(+:sum_val) private(i) shared(a,b,c,d,e)
  do i = 1, LEN_1D
    a(i) = c(i) + d(i)
    sum_val = sum_val + a(i)
    b(i) = c(i) + e(i)
    sum_val = sum_val + b(i)
  end do

  b(1) = sum_val

end subroutine tsvc_2_s319_fp64
