module tsvc_2_s311_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: sum_out(*)
    integer(c_int64_t), value :: LEN_1D
    real(c_double) :: sum
    integer(c_int64_t) :: i
    sum = 0.0_c_double
    !$omp parallel do reduction(+:sum) schedule(static)
    do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
      sum = sum + a(i + 1_c_int64_t)
    end do
    !$omp end parallel do
    sum_out(1) = sum
  end subroutine tsvc_2_s311_fp64
end module tsvc_2_s311_mod
