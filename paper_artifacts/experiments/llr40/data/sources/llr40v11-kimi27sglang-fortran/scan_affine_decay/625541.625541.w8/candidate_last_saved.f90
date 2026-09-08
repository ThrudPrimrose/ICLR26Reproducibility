subroutine scan_affine_decay_fp64(y, c, x, n, workspace, workspace_bytes) bind(c, name='scan_affine_decay_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
  real(c_double), intent(inout) :: y(*)
  real(c_double), intent(in) :: c(*), x(*)
  integer(c_int64_t), value, intent(in) :: n
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_bytes

  integer(c_int64_t) :: i

  do i = 2_c_int64_t, n
    y(i) = c(i) * y(i-1) + x(i)
  end do
end subroutine scan_affine_decay_fp64
