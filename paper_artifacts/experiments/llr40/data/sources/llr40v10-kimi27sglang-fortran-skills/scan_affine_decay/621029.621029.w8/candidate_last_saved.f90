subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: y(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), x(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, n

  n = LEN_1D
  if (n <= 1) return
  do i = 2, n
    y(i) = c(i) * y(i - 1) + x(i)
  end do
end subroutine scan_affine_decay_fp64
