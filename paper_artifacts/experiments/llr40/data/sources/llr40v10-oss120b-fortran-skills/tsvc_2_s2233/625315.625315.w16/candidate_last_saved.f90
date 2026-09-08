subroutine tsvc_2_s2233_fp64(a, b, c, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: c(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  ! Compute a: recurrence over second index (j) for each column i
  !$omp parallel do default(none) shared(a,c,LEN_2D) private(i,j) schedule(static)
  do i = 9, LEN_2D
    do j = 9, LEN_2D
      a(i, j) = a(i, j-1) + c(i, j)
    end do
  end do
  !$omp end parallel do

  ! Compute b: recurrence over first index (i) for each row j; parallelize outer j loop
  !$omp parallel do default(none) shared(b,c,LEN_2D) private(i,j) schedule(static)
  do i = 9, LEN_2D
    do j = 9, LEN_2D
      b(i, j) = b(i, j-1) + c(i, j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s2233_fp64
