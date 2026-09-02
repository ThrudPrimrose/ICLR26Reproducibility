subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d, workspace, workspace_size) &
     bind(C, name="tsvc_2_s152_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i

!$omp parallel do
  do i = 1, len_1d
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
  if (workspace_size > 0) workspace(1) = workspace(1)
end subroutine tsvc_2_s152_fp64
