subroutine tsvc_2_s3112_fp64(a, b, len_1d, workspace, workspace_size) bind(C, name="tsvc_2_s3112_fp64")
  use iso_c_binding
  implicit none

  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  type(c_ptr), value, intent(in) :: workspace

  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(len_1d)

  real(c_double) :: sum
  integer(c_int64_t) :: i

  sum = 0.0_c_double
  do i = 1, len_1d
    sum = sum + a(i)
    b(i) = sum
  end do

end subroutine tsvc_2_s3112_fp64
