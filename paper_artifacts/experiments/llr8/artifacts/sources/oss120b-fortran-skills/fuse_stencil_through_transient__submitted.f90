! Fortran implementation of fuse_stencil_through_transient kernel
! Computes out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
! for i = 2 .. LEN_1D-2 (1-based indexing)

subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: out(LEN_1D)
  integer(c_int64_t) :: i

  ! Parallelize the loop; each iteration is independent.
  !$omp parallel do simd schedule(static)
  do i = 2, LEN_1D-2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
end subroutine

