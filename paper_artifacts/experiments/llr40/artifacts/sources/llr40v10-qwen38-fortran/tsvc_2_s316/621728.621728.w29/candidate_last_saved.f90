subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name='tsvc_2_s316_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: result(1)
  integer(c_int64_t), value :: len_1d
  real(c_double) :: x
  integer(c_int64_t) :: i
  x = a(1)
!$omp parallel do reduction(min: x)
  do i = 2, len_1d
    x = min(x, a(i))
  end do
!$omp end parallel do
  result(1) = x
end subroutine
