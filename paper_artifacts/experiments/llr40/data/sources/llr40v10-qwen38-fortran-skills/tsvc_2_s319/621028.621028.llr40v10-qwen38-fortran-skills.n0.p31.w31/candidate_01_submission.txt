subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  integer(c_int64_t) :: i
  real(c_double) :: sum_val

  sum_val = 0.0d0
  !$omp parallel do simd reduction(+:sum_val)
  do i = 1, len_1d
    a(i) = c(i) + d(i)
    sum_val = sum_val + a(i)
    b(i) = c(i) + e(i)
    sum_val = sum_val + b(i)
  end do
  b(1) = sum_val
end subroutine tsvc_2_s319_fp64
