subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name='fuse_stencil_through_transient_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(0:LEN_1D-1)
  real(c_double), intent(out) :: out(0:LEN_1D-1)
  integer(c_int64_t) :: i

  !GCC$ ivdep
  !GCC$ unroll 16
  do i = 1, LEN_1D - 3
    out(i) = (a(i - 1) + a(i) + a(i + 1)) * (a(i) + a(i + 1) + a(i + 2))
  end do
end subroutine fuse_stencil_through_transient_fp64
