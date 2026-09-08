subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: t
  !$omp parallel do
  do i = 1, LEN_1D
    t = a(i) * a(i)
    out(i) = (t + 1.0_c_double) * (t - 1.0_c_double)
  end do
  !$omp end parallel do
end subroutine fuse_diamond_fp64
