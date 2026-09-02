subroutine fuse_diamond_fp64(a, out, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(inout) :: out(len_1d)
  character(kind=c_char), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i

  ! Reference: t(i) = a(i)*a(i); u(i) = t(i)+1; v(i) = t(i)-1; out(i) = u(i)*v(i)
  ! Fully fused: one read of a, one write of out. Same per-element op order
  ! as the reference, so bit-identical results.
  !$omp parallel do simd schedule(static)
  do i = 1, len_1d
    out(i) = (a(i) * a(i) + 1.0d0) * (a(i) * a(i) - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
