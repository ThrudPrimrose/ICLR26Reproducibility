subroutine tsvc_2_s4112_fp64(a, b, ip, n) bind(C)
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n)
  integer(c_int32_t), intent(in) :: ip(n)
  integer :: i
  !$omp simd
  do i = 1, n
    a(i) = a(i) + 2.0d0 * b(ip(i))
  end do
  !$omp end simd
end subroutine tsvc_2_s4112_fp64
