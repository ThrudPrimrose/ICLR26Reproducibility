module fuse_stencil_through_transient_m
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(c,name='fuse_stencil_through_transient_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: out(LEN_1D)
    integer(c_int64_t) :: i
    !$omp parallel do simd schedule(static) aligned(a:64,out:64)
    do i = 2_c_int64_t, LEN_1D - 2_c_int64_t
      out(i) = (a(i-1_c_int64_t) + a(i) + a(i+1_c_int64_t)) &
             * (a(i) + a(i+1_c_int64_t) + a(i+2_c_int64_t))
    end do
    !$omp end parallel do simd
  end subroutine fuse_stencil_through_transient_fp64
end module fuse_stencil_through_transient_m
