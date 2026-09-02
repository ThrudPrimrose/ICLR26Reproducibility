subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(c)
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(*), b(*)
  real(c_double), intent(in) :: c(*), d(*), e(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double) :: sum
  integer(c_int64_t) :: i

  sum = 0.0_c_double
  !$omp parallel do simd reduction(+:sum) schedule(static) proc_bind(close) default(none) shared(a,b,c,d,e,LEN_1D) private(i)
  do i = 1, LEN_1D
    a(i) = c(i) + d(i)
    sum = sum + a(i)
    b(i) = c(i) + e(i)
    sum = sum + b(i)
  end do
  !$omp end parallel do simd
  b(1) = sum
end subroutine tsvc_2_s319_fp64
