subroutine fuse_stencil_through_transient_fp64(a, out, len_1d, workspace, workspace_size) bind(C)
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(inout) :: out(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer :: i

  ! Fused pointwise form:
  !   tmp(i)   = a(i-1) + a(i) + a(i+1)
  !   out(i)   = tmp(i) * tmp(i+1),  i = 2 .. len_1d-2
  ! so  out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  ! Fully parallel, unit stride, no carried state.
  !$omp parallel do simd num_threads(48)
  do i = 2, len_1d - 2
    out(i) = (a(i - 1) + a(i) + a(i + 1)) * (a(i) + a(i + 1) + a(i + 2))
  end do
end subroutine fuse_stencil_through_transient_fp64
