subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d, workspace_size
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in)    :: b(len_2d)
  real(c_double), intent(in)    :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: c(len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)
  real(c_double), intent(in)    :: d(len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j

  !$omp parallel do
  do j = 1, len_2d
    do i = 1, len_2d
      aa(i, j) = aa(i, j) + bb(i, j) * cc(i, j)
    end do
  end do

  !$omp parallel do simd
  do i = 1, len_2d
    a(i) = b(i) + c(i) * d(i)
  end do
end subroutine tsvc_2_s2275_fp64
