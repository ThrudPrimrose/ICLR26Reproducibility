subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  integer(c_int64_t) :: i

  if (LEN_1D <= 0) return

  ! First element: a[0] = b[0] * c[0], no previous product
  a(1) = b(1) * c(1)

  ! Parallel loop for remaining elements
  !$omp parallel do
  do i = 2, LEN_1D
    a(i) = b(i) * c(i) + b(i-1) * c(i-1)
  end do
  !$omp end parallel do

end subroutine tsvc_2_s252_fp64
