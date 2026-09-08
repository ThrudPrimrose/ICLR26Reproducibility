subroutine wf_triangular_fp64(a, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, workspace_size
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j

  do i = 2, LEN_2D
    do j = i, LEN_2D
      a(j, i) = a(j, i) + a(j, i - 1) + a(j - 1, i)
    end do
  end do
end subroutine wf_triangular_fp64
