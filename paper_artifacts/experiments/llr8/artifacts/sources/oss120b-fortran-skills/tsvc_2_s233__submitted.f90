subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d, workspace, workspace_size) bind(C, name="tsvc_2_s233_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j

  ! Compute aa: vertical cumulative sum (depends on previous row) across columns i.
  !$omp parallel do default(none) private(i, j) shared(aa, cc, len_2d)
  do i = 9, len_2d
    do j = 9, len_2d
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
  end do
  !$omp end parallel do

  ! Compute bb: horizontal cumulative sum (depends on previous row) across columns j.
  !$omp parallel do default(none) private(i, j) shared(bb, cc, len_2d)
  do j = 9, len_2d
    do i = 9, len_2d
      bb(i, j) = bb(i - 1, j) + cc(i, j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s233_fp64
