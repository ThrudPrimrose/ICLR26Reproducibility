subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, vlen, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j, jmax

  !$omp parallel do schedule(dynamic, 16)
  do i = 1, len_2d
    jmax = (i - 1) / vlen + 1
    !$omp simd
    do j = 1, jmax
      aa(j, i) = bb(j, i) + cc(j, i)
    end do
  end do
end subroutine tsvc_2_s1232_fp64
