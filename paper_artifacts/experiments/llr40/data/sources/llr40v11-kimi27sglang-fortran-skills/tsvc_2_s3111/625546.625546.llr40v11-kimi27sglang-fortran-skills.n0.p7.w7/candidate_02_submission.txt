subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(2)
  integer(c_int64_t) :: i
  real(c_double) :: sum

  sum = 0.0_c_double
  !$omp parallel do simd reduction(+:sum) schedule(static)
  do i = 1, LEN_1D
    sum = sum + max(a(i), 0.0_c_double)
  end do
  b(1) = sum
end subroutine tsvc_2_s3111_fp64
