subroutine fuse_diamond_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  real(c_double) :: x
  !$omp parallel do simd schedule(static)
  do i = 1, LEN_1D
    x = a(i)
    out(i) = x * x * x * x - 1.0d0
  end do
end subroutine fuse_diamond_fp64
