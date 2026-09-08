module fuse_stencil_through_transient_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name='fuse_stencil_through_transient_fp64')
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: out(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: s1, s2

    !$omp parallel do simd schedule(static) private(i, s1, s2)
    !GCC$ unroll 4
    do i = 2, LEN_1D - 2
       s1 = a(i-1) + a(i) + a(i+1)
       s2 = a(i)   + a(i+1) + a(i+2)
       out(i) = s1 * s2
    end do
  end subroutine fuse_stencil_through_transient_fp64
end module fuse_stencil_through_transient_mod
