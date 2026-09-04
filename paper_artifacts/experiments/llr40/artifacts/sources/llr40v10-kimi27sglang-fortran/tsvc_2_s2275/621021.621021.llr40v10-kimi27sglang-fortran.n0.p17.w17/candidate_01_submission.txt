subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(c, name="tsvc_2_s2275_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(out) :: a(0:LEN_2D-1)
  real(c_double), intent(inout) :: aa(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: b(0:LEN_2D-1)
  real(c_double), intent(in) :: bb(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: c(0:LEN_2D-1)
  real(c_double), intent(in) :: cc(0:LEN_2D*LEN_2D-1)
  real(c_double), intent(in) :: d(0:LEN_2D-1)
  integer(c_int64_t) :: i, j, base

  !$omp parallel do schedule(static) private(i, base)
  do j = 0, LEN_2D - 1
    base = j * LEN_2D
    !$omp simd
    do i = 0, LEN_2D - 1
      aa(base + i) = aa(base + i) + bb(base + i) * cc(base + i)
    end do
  end do
  !$omp end parallel do

  !$omp parallel do simd schedule(static)
  do i = 0, LEN_2D - 1
    a(i) = b(i) + c(i) * d(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s2275_fp64
