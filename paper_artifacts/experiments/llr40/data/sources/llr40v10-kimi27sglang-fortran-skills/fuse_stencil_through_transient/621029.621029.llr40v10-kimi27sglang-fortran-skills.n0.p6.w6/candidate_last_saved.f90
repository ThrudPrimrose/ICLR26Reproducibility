subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: t1, t2

  !$omp parallel do simd
  do i = 2, LEN_1D - 2
    t1 = a(i - 1) + a(i) + a(i + 1)
    t2 = a(i) + a(i + 1) + a(i + 2)
    out(i) = t1 * t2
  end do
  !$omp end parallel do simd
end subroutine fuse_stencil_through_transient_fp64
