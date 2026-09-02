subroutine fuse_diamond_fp64(a, out, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(n)
  real(c_double), intent(out) :: out(n)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: i
  real(c_double) :: t
  !$omp parallel do simd private(t)
  do i = 1, n
    t = a(i) * a(i)
    out(i) = (t + 1.0d0) * (t - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
