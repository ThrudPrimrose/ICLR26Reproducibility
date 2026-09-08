subroutine fuse_diamond_fp64(a, out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(len_1d)
  integer(c_int64_t) :: i, n, rem
  real(c_double) :: t0, t1, t2, t3

  n = len_1d
  rem = iand(n, 3_c_int64_t)
  n = n - rem

  !$omp parallel do simd schedule(static)
  do i = 1, n, 4
    t0 = a(i) * a(i)
    t1 = a(i + 1) * a(i + 1)
    t2 = a(i + 2) * a(i + 2)
    t3 = a(i + 3) * a(i + 3)
    out(i) = (t0 + 1.0d0) * (t0 - 1.0d0)
    out(i + 1) = (t1 + 1.0d0) * (t1 - 1.0d0)
    out(i + 2) = (t2 + 1.0d0) * (t2 - 1.0d0)
    out(i + 3) = (t3 + 1.0d0) * (t3 - 1.0d0)
  end do
  !$omp end parallel do simd

  do i = n + 1, len_1d
    t0 = a(i) * a(i)
    out(i) = (t0 + 1.0d0) * (t0 - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
