subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(c, name="tsvc_2_s3112_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: b(*)
  integer(c_int64_t), value :: len_1d
  integer(c_int64_t) :: i
  real(c_double) :: s
  s = 0.0d0
  do i = 1, len_1d
    s = s + a(i)
    b(i) = s
  end do
end subroutine tsvc_2_s3112_fp64
