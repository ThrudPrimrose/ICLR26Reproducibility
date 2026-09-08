subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: b(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i
  real(c_double) :: sum
  if (LEN_1D <= 0) return
  sum = 0.0_c_double
  do i = 1_c_int64_t, LEN_1D
    sum = sum + a(i)
    b(i) = sum
  end do
end subroutine tsvc_2_s3112_fp64
