! Affine scan with a variable coefficient: y(i) = c(i)*y(i-1) + x(i), i = 2..n.
! y(1) is the seed (already set). C-ABI symbol per the judge binding:
!   void scan_affine_decay_fp64(double *c, double *x, double *y, int64_t LEN_1D,
!                               void *workspace, int64_t workspace_size);
subroutine scan_affine_decay_fp64(c, x, y, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: x(len_1d)
  real(c_double), intent(inout) :: y(len_1d)
  type(c_ptr), value, intent(in) :: workspace

  integer(c_int64_t) :: i
  real(c_double) :: v

  if (len_1d <= 1) return
  v = y(1)
  do i = 2, len_1d
    y(i) = c(i) * v + x(i)
    v = y(i)
  end do
end subroutine
