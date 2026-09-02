subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s1244_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
  real(c_double), intent(inout) :: a(*), d(*)
  real(c_double), intent(in) :: b(*), c(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  do i = 1, LEN_1D - 1
    a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    d(i) = a(i) + a(i + 1)
  end do
end subroutine tsvc_2_s1244_fp64
