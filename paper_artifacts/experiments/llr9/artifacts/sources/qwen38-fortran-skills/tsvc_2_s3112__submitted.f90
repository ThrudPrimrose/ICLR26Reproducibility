subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
  use, intrinsic :: iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: s
  s = 0.0d0
  do i = 1, LEN_1D
    s = s + a(i)
    b(i) = s
  end do
end subroutine tsvc_2_s3112_fp64
