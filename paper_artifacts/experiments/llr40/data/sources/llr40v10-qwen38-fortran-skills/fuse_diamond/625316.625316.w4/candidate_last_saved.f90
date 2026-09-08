subroutine fuse_diamond_fp64(a, out, len_1d) bind(C, name="fuse_diamond_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(len_1d)

  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, len_1d
    out(i) = (a(i) * a(i) + 1.0d0) * (a(i) * a(i) - 1.0d0)
  end do
end subroutine fuse_diamond_fp64
