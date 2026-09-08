subroutine tsvc_2_s119_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i, j

  do j = 2, len_2d
    do i = 2, len_2d
      aa(i, j) = aa(i - 1, j - 1) + bb(i, j)
    end do
  end do
end subroutine tsvc_2_s119_fp64
