subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int64_t) :: i

  if (LEN_1D < 1) return
  ! a(1) = b(1)*c(1) + 0.0  (t starts at 0.0)
  a(1) = b(1) * c(1)
  if (LEN_1D < 2) return
  ! rotated scalar: t at iteration i is exactly s(i-1) = b(i-1)*c(i-1),
  ! so a(i) = b(i)*c(i) + b(i-1)*c(i-1)  -- elementwise, bit-identical to serial
  !$omp parallel do simd
  do i = 2, LEN_1D
    a(i) = b(i) * c(i) + b(i - 1) * c(i - 1)
  end do
end subroutine tsvc_2_s252_fp64
