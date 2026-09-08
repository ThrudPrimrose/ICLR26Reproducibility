subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_2D, workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)

  integer(c_int64_t) :: i, j

  !$omp parallel do private(i, j) schedule(static)
  do i = 9, LEN_2D
    do j = 9, LEN_2D
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
  end do
  !$omp end parallel do

  !$omp parallel do private(i, j) schedule(static)
  do j = 9, LEN_2D
    do i = 9, LEN_2D
      bb(j, i) = bb(j, i - 1) + cc(j, i)
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s2233_fp64
