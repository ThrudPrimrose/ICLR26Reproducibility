! Optimized fused diamond kernel for double precision (fp64) with fused multiplication
subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name="fuse_diamond_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: out(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: t
  ! Compute out = a^4 - 1 using two multiplications per element.
  do i = 1, LEN_1D
    t = a(i) * a(i)
    out(i) = t * t - 1.0d0
  end do
end subroutine fuse_diamond_fp64
