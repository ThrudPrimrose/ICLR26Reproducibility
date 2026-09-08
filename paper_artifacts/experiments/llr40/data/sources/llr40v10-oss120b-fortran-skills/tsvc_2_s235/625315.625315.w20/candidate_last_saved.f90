subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D)
  real(c_double), intent(in) :: b(LEN_2D), c(LEN_2D)
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  ! Update a
  !$omp parallel do
  do i = 1, LEN_2D
    a(i) = a(i) + b(i) * c(i)
  end do
  !$omp end parallel do

  ! Compute aa using transformed loops for better memory access
  !$omp parallel private(i, j)
  do j = 2, LEN_2D
    !$omp do
    do i = 1, LEN_2D
      aa(i, j) = aa(i, j-1) + bb(i, j) * a(i)
    end do
    !$omp end do
  end do
  !$omp end parallel

end subroutine tsvc_2_s235_fp64
