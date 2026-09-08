subroutine fuse_stencil_through_transient_fp64(a, out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(len_1d)

  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 2, len_1d - 2
    out(i) = (a(i - 1) + a(i) + a(i + 1)) * (a(i) + a(i + 1) + a(i + 2))
  end do
end subroutine fuse_stencil_through_transient_fp64
