module fuse_stencil_through_transient_mod
  use iso_c_binding
  implicit none
contains
  subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: sum1, sum2
    
    !$omp parallel do simd default(none) schedule(static) shared(a, out, LEN_1D) private(i, sum1, sum2)
    do i = 2, LEN_1D-2
      sum1 = a(i-1) + a(i) + a(i+1)
      sum2 = a(i) + a(i+1) + a(i+2)
      out(i) = sum1 * sum2
    end do
    !$omp end parallel do simd
  end subroutine fuse_stencil_through_transient_fp64
end module fuse_stencil_through_transient_mod
