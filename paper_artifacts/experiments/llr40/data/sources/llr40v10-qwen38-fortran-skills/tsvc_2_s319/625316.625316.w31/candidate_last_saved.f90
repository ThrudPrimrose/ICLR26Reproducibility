subroutine tsvc_2_s319(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s319_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  real(c_double) :: sum, ai, bi
  integer(c_int64_t) :: i

  sum = 0.0d0
  !$omp parallel do simd reduction(+:sum)
  do i = 1, len_1d
    ai = c(i) + d(i)
    bi = c(i) + e(i)
    a(i) = ai
    b(i) = bi
    sum = sum + ai + bi
  end do
  b(1) = sum
end subroutine tsvc_2_s319
