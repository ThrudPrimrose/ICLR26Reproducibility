subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(out)   :: a(LEN_1D)
  real(c_double), intent(out)   :: b(LEN_1D)
  real(c_double), intent(in)    :: c(LEN_1D)
  real(c_double), intent(in)    :: d(LEN_1D)
  real(c_double), intent(in)    :: e(LEN_1D)
  integer :: i, n32
  integer(c_int64_t) :: i64
  real(c_double) :: sum, ai, bi

  sum = 0.0_c_double
  if (LEN_1D <= 2147483647_c_int64_t) then
    n32 = int(LEN_1D)
    !$omp parallel do simd reduction(+:sum) private(ai,bi)
    do i = 1, n32
      ai = c(i) + d(i)
      bi = c(i) + e(i)
      a(i) = ai
      b(i) = bi
      sum = sum + ai + bi
    end do
  else
    !$omp parallel do reduction(+:sum) private(ai,bi)
    do i64 = 1, LEN_1D
      ai = c(i64) + d(i64)
      bi = c(i64) + e(i64)
      a(i64) = ai
      b(i64) = bi
      sum = sum + ai + bi
    end do
  end if
  if (LEN_1D > 0_c_int64_t) b(1) = sum
end subroutine tsvc_2_s319_fp64
