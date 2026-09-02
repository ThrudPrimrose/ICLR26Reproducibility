subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C, name="tsvc_2_s231_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, *)
  real(c_double), intent(in) :: bb(LEN_2D, *)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i, j

  !$omp parallel do default(none) shared(aa, bb, LEN_2D) private(i, j)
  do i = 1, LEN_2D
    do j = 2, LEN_2D
      aa(i, j) = aa(i, j-1) + bb(i, j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s231_fp64