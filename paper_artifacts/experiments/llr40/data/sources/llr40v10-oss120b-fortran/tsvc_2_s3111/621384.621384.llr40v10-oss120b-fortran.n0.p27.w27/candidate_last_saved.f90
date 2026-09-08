subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3111_fp64")
  use iso_c_binding
  implicit none
  ! Arguments
  real(C_DOUBLE), intent(in) :: a(*)
  real(C_DOUBLE), intent(out) :: b(*)
  integer(C_INT64_T), value, intent(in) :: LEN_1D
  ! Locals
  real(C_DOUBLE) :: sum
  integer(C_INT64_T) :: i

  sum = 0.0_C_DOUBLE
  !$omp parallel do reduction(+:sum) schedule(static)
  do i = 0, LEN_1D-1
    if (a(i+1) > 0.0_C_DOUBLE) then
      sum = sum + a(i+1)
    end if
  end do
  !$omp end parallel do
  b(1) = sum
end subroutine tsvc_2_s3111_fp64
