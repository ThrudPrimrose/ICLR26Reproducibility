subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: sum_out(*)
  integer(c_int64_t) :: i
  real(c_double) :: s

  s = 0.0_c_double
!$omp parallel do reduction(+:s) schedule(static)
  do i = 1, LEN_1D
    s = s + a(i)
  end do
!$omp end parallel do

  sum_out(1) = s
end subroutine tsvc_2_s311_fp64
