subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d, workspace, workspace_bytes) bind(C, name="tsvc_2_s323_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(inout) :: b(*)
  real(c_double), intent(in) :: c(*)
  real(c_double), intent(in) :: d(*)
  real(c_double), intent(in) :: e(*)
  integer(c_int64_t), value :: len_1d
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes

  integer(c_int64_t) :: i

  if (len_1d <= 1) return

  do i = 2, len_1d
    a(i) = b(i-1) + c(i) * d(i)
    b(i) = a(i) + c(i) * e(i)
  end do
end subroutine tsvc_2_s323_fp64
