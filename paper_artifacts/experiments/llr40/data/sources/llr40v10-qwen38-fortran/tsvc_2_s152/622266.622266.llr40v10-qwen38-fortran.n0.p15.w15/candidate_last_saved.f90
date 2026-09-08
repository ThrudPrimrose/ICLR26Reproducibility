subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(C, name='tsvc_2_s152_fp64')
  implicit none
  real(kind=8), intent(inout) :: a(*)
  real(kind=8), intent(inout) :: b(*)
  real(kind=8), intent(in)    :: c(*)
  real(kind=8), intent(in)    :: d(*)
  real(kind=8), intent(in)    :: e(*)
  integer(kind=8), intent(in), value :: LEN_1D

  integer(kind=8) :: i

  if (LEN_1D <= 0) return
  !$omp parallel do schedule(static)
  do i = 1, LEN_1D
     b(i) = d(i) * e(i)
     a(i) = a(i) + b(i) * c(i)
  end do
  !$omp end parallel do
end subroutine tsvc_2_s152_fp64
