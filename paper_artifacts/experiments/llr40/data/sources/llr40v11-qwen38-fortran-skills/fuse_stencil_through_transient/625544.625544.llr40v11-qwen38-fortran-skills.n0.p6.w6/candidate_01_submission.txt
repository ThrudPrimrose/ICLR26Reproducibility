subroutine fuse_stencil_through_transient_fp64(a, out, len_1d) bind(C)
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: out(len_1d)
  integer(c_int64_t) :: i
  if (len_1d < 40000) then
    do i = 2, len_1d - 2
      out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
  else
    !$omp parallel do simd schedule(static)
    do i = 2, len_1d - 2
      out(i) = (a(i-1) + a(i) + a(i+1)) * (a(i) + a(i+1) + a(i+2))
    end do
  end if
end subroutine fuse_stencil_through_transient_fp64
