! Fortran implementation of fuse_stencil_through_transient kernel
! Computes out(i) = (a(i-1)+a(i)+a(i+1)) * (a(i)+a(i+1)+a(i+2))
! for i = 2 .. LEN_1D-2 (Fortran 1-based indexing), matching the C reference.

subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: out(LEN_1D)
  integer(c_int64_t) :: i

  ! Parallelize and vectorize the main computation.
  !$omp parallel do simd default(none) shared(a, out, LEN_1D) private(i)
  do i = 2, LEN_1D-2
    out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
  end do

end subroutine fuse_stencil_through_transient_fp64
