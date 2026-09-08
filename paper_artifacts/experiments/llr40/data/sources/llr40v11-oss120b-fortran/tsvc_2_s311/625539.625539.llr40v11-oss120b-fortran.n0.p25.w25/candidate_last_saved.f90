module tsvc_2_s311_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    implicit none
    real(C_DOUBLE), intent(in) :: a(*)
    real(C_DOUBLE), intent(out) :: sum_out(*)
    integer(C_INT64_T), value :: LEN_1D
    real(C_DOUBLE) :: sum
    integer(C_INT64_T) :: i
    sum = 0.0_C_DOUBLE
    !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(+:sum)
    do i = 1, LEN_1D
      sum = sum + a(i)
    end do
    !$omp end parallel do
    sum_out(1) = sum
  end subroutine tsvc_2_s311_fp64
end module tsvc_2_s311_mod
