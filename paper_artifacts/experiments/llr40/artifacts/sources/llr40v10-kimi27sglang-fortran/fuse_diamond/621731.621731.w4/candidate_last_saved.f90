subroutine fuse_diamond_fp64(out, a, n, workspace, ws_size) bind(c, name='fuse_diamond_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
  real(c_double), intent(out) :: out(*)
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), value, intent(in) :: n
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: ws_size

  integer(c_int64_t) :: i
  real(c_double) :: t

  !$omp parallel do simd default(none) private(i, t) shared(a, out, n)
  do i = 1, n
    t = a(i) * a(i)
    out(i) = (t + 1.0_c_double) * (t - 1.0_c_double)
  end do
  !$omp end parallel do simd
end subroutine fuse_diamond_fp64
