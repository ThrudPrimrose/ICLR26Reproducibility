module fuse_stencil_through_transient_m
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(c, name='fuse_stencil_through_transient_fp64')
    implicit none
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i
    integer(c_int64_t) :: n
    n = LEN_1D - 2_c_int64_t
    !$omp parallel do simd schedule(static)
    do i = 2_c_int64_t, n
       out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
    !$omp end parallel do simd
  end subroutine fuse_stencil_through_transient_fp64
end module fuse_stencil_through_transient_m
