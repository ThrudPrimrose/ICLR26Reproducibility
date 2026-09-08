subroutine fuse_diamond_fp64(a, out, len1d, workspace, workspace_size) bind(C, name="fuse_diamond_fp64")
  use, intrinsic :: iso_c_binding
  integer(c_int64_t), value, intent(in) :: len1d, workspace_size
  type(c_ptr), intent(in) :: workspace
  real(c_double), intent(in)  :: a(len1d)
  real(c_double), intent(out) :: out(len1d)
  integer(c_int64_t) :: i
  double precision :: t

  !$omp parallel do simd
  do i = 1, len1d
    t = a(i) * a(i)
    out(i) = (t + 1.0d0) * (t - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
