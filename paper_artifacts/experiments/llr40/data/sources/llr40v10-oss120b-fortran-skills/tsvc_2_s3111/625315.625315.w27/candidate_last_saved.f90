subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(1)
  integer(c_int64_t) :: i
  real(c_double) :: sum

  sum = 0.0_c_double
  !$omp parallel do reduction(+:sum) schedule(static)
  do i = 1, LEN_1D
    if (a(i) > 0.0_c_double) then
      sum = sum + a(i)
    end if
  end do
  !$omp end parallel do
  b(1) = sum
end subroutine tsvc_2_s3111_fp64
