subroutine wf_triangular_fp64(a, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size / 8)
  integer(c_int64_t) :: i, j

  do i = 2, len_2d
    do j = i, len_2d
      a(j, i) = a(j, i) + a(j, i-1) + a(j-1, i)
    end do
  end do
end subroutine wf_triangular_fp64
