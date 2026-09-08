subroutine fuse_diamond_fp64(a, out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(len_1d)
  integer(c_int64_t) :: n, i
  real(c_double) :: t

  n = len_1d
  !$omp parallel do schedule(static) default(none) shared(n, a, out) private(t)
  do i = 1, n
     t = a(i) * a(i)
     out(i) = (t + 1.0d0) * (t - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
