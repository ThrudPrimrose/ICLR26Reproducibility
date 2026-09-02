module tsvc_2_s3111_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(c)
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(inout) :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: sum_val
    integer(c_int64_t) :: i

    sum_val = 0.0_c_double
    !$omp parallel do reduction(+:sum_val)
    do i = 1, LEN_1D
      if (a(i) > 0.0_c_double) then
        sum_val = sum_val + a(i)
      end if
    end do
    !$omp end parallel do
    b(1) = sum_val
  end subroutine tsvc_2_s3111_fp64
end module tsvc_2_s3111_mod
