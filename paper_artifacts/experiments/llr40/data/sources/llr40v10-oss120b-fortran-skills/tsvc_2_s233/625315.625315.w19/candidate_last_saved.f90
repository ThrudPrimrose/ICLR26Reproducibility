subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s233_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  ! Compute aa: column-wise recurrence, parallel over columns (i)
!$omp parallel do private(j) schedule(static)
  do i = 9, LEN_2D
    do j = 9, LEN_2D
      aa(j,i) = aa(j-1,i) + cc(j,i)
    end do
  end do
!$omp end parallel do

  ! Compute bb: row-wise recurrence, parallel over rows (j)
!$omp parallel do private(i) schedule(static)
  do j = 9, LEN_2D
    do i = 9, LEN_2D
      bb(j,i) = bb(j,i-1) + cc(j,i)
    end do
  end do
!$omp end parallel do

end subroutine tsvc_2_s233_fp64
