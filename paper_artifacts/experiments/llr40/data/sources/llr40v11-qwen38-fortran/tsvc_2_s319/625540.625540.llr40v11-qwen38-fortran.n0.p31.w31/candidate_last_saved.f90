subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s319_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a, b
  real(c_double), dimension(*), intent(in) :: c, d, e
  integer(c_int64_t), value, intent(in) :: len_1d

  real(c_double) :: sum
  integer(c_int64_t) :: i

  sum = 0.0d0
!$omp parallel do reduction(+:sum) schedule(static)
  do i = 1, len_1d
    a(i) = c(i) + d(i)
    sum = sum + a(i)
    b(i) = c(i) + e(i)
    sum = sum + b(i)
  end do
!$omp end parallel do
  if (len_1d > 0) b(1) = sum
end subroutine tsvc_2_s319_fp64
