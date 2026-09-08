subroutine tsvc_2_s319(a, b, c, d, e, LEN_1D) bind(C, name='tsvc_2_s319_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in)    :: c(LEN_1D), d(LEN_1D), e(LEN_1D)

  real(c_double) :: sum_val, ci, ai, bi
  integer(c_int64_t) :: i

  sum_val = 0.0_c_double

  !$omp parallel do reduction(+:sum_val) schedule(static) private(ci, ai, bi)
  do i = 1, LEN_1D
    ci = c(i)
    ai = ci + d(i)
    bi = ci + e(i)
    a(i) = ai
    b(i) = bi
    sum_val = sum_val + ai + bi
  end do
  !$omp end parallel do

  b(1) = sum_val
end subroutine tsvc_2_s319
