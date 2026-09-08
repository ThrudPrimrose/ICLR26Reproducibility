subroutine scan_affine_decay_fp64(y, c, x, LEN_1D) bind(C, name='scan_affine_decay_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: y(*)
  real(c_double), intent(in) :: c(*)
  real(c_double), intent(in) :: x(*)
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int64_t) :: i

  if (LEN_1D <= 0) return
  y(1) = x(1)
  do i = 2, LEN_1D
    y(i) = c(i) * y(i - 1) + x(i)
  end do
end subroutine scan_affine_decay_fp64
