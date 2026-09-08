subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D, workspace, workspace_size) bind(C, name="tsvc_2_s2233_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  ! Compute aa recurrences - parallel over i (outer loop)
  !$omp parallel do default(none) shared(aa, cc, LEN_2D) private(i, j) schedule(static)
  do i = 9, LEN_2D
    do j = 9, LEN_2D
      aa(i,j) = aa(i,j-1) + cc(i,j)
    end do
  end do
  !$omp end parallel do

  ! Compute bb recurrences - parallel over j (outer loop)
  !$omp parallel do default(none) shared(bb, cc, LEN_2D) private(i, j) schedule(static)
  do j = 9, LEN_2D
    do i = 9, LEN_2D
      bb(i,j) = bb(i,j-1) + cc(i,j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s2233_fp64
