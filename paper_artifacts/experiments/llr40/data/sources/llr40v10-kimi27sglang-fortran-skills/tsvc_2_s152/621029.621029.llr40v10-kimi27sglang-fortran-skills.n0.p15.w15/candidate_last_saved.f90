subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer :: i

  !$omp parallel do simd schedule(static) proc_bind(close)
  do i = 1, int(LEN_1D)
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s152_fp64
