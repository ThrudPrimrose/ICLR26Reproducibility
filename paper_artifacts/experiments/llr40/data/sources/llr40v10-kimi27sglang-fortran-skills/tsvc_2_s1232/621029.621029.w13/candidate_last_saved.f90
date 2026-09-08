subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN, workspace, workspace_size) bind(C, name='tsvc_2_s1232_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D, VLEN, workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  !$omp parallel do simd
  do j = 1, LEN_2D
    do i = j * VLEN + 1, LEN_2D
      aa(i, j) = bb(i, j) + cc(i, j)
    end do
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s1232_fp64
