subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(c, name='tsvc_2_s2710_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: gt, a_new, b_new, c_true, c_false
  logical :: cond_len, cond_x

  cond_len = (LEN_1D > 10_c_int64_t)
  cond_x = (x(1) > 0.0_c_double)

  !$omp simd simdlen(8) private(gt, a_new, b_new, c_true, c_false)
  do i = 1_c_int64_t, LEN_1D
     gt = merge(1.0_c_double, 0.0_c_double, a(i) > b(i))
     a_new = a(i) + b(i) * d(i)
     b_new = a(i) + e(i) * e(i)
     c_true = merge(c(i) + d(i) * d(i), d(i) * e(i) + 1.0_c_double, cond_len)
     c_false = merge(a(i) + d(i) * d(i), c(i) + e(i) * e(i), cond_x)
     a(i) = gt * a_new + (1.0_c_double - gt) * a(i)
     b(i) = gt * b(i) + (1.0_c_double - gt) * b_new
     c(i) = gt * c_true + (1.0_c_double - gt) * c_false
  end do
  !$omp end simd
end subroutine tsvc_2_s2710_fp64
