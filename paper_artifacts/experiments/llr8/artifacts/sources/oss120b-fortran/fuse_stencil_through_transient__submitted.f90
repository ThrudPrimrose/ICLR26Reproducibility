module fuse_stencil_mod
  use iso_c_binding
  implicit none
contains

  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: a(0:*)
    real(c_double), intent(out) :: out(0:*)
    integer(c_int64_t) :: i

    ! Parallelize outer loop with SIMD vectorization
    !$omp parallel do schedule(static) default(none) shared(a, out, LEN_1D) private(i)
    do i = 1, LEN_1D - 3
      out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
    !$omp end parallel do
  end subroutine fuse_stencil_through_transient_fp64

end module fuse_stencil_mod
