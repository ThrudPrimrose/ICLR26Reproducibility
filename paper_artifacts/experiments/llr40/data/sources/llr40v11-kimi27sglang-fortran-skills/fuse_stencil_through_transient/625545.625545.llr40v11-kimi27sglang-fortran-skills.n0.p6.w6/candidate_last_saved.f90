subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in)  :: a(LEN_1D)
  real(c_double), intent(out) :: out(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: s1, s2

  !$omp parallel do simd schedule(static) default(none) shared(a, out, LEN_1D) private(s1, s2)
  do i = 2, LEN_1D - 2
     s1 = a(i - 1) + a(i) + a(i + 1)
     s2 = a(i) + a(i + 1) + a(i + 2)
     out(i) = s1 * s2
  end do
  !$omp end parallel do simd
end subroutine fuse_stencil_through_transient_fp64
