subroutine scan_affine_decay_fp64(c, x, y, n, workspace, workspace_bytes) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_bytes
  real(c_double), intent(in) :: c(n), x(n)
  real(c_double), intent(inout) :: y(n)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: i

  if (n <= 0) return
  y(1) = x(1)
  do i = 2, n
     y(i) = c(i) * y(i-1) + x(i)
  end do
end subroutine scan_affine_decay_fp64
