subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name='fuse_stencil_through_transient_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(in)  :: a
  real(c_double), dimension(*), intent(out) :: out
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int64_t) :: i

  !$omp parallel do schedule(static)
  do i = 2, LEN_1D - 2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do
  !$omp end parallel do
end subroutine fuse_stencil_through_transient_fp64
