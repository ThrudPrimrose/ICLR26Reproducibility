subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: result

  real(c_double) :: x
  integer(c_int64_t) :: i

  x = a(1)
  !$omp parallel do simd reduction(min:x)
  do i = 2, LEN_1D
    x = min(x, a(i))
  end do
  !$omp end parallel do simd

  result = x
end subroutine tsvc_2_s316_fp64
