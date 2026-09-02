subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C)
  use iso_c_binding, only: c_int64_t, c_double
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)
  real(c_double) :: x
  integer :: i, n

  n = int(len_1d)
  x = a(1)
  !$omp parallel do simd reduction(min:x) schedule(static)
  do i = 2, n
    x = min(x, a(i))
  end do
  !$omp end parallel do simd
  result(1) = x
end subroutine tsvc_2_s316_fp64
