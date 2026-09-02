subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  real(c_double), intent(out) :: d(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, LEN_1D - 1
    d(i) = a(i + 1)
  end do
  !$omp end parallel do simd

  !$omp parallel do simd
  do i = 1, LEN_1D - 1
    a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    d(i) = a(i) + d(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s1244_fp64
