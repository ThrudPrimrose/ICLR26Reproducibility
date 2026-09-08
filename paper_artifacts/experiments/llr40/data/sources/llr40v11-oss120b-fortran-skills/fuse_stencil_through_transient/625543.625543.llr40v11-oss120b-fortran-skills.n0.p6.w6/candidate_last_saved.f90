subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(LEN_1D)
  integer(c_int64_t) :: i

  if (LEN_1D <= 3_c_int64_t) return

  !$omp parallel default(none) shared(a, out, LEN_1D) private(i)
    !$omp do schedule(static)
    do i = 2, LEN_1D-2
      out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
    !$omp end do
  !$omp end parallel

end subroutine fuse_stencil_through_transient_fp64
