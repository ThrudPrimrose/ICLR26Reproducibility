module tsvc_2_s311_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    ! Arguments: a - input array, sum_out - output scalar, LEN_1D - length of a
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: sum_out(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: sum_local

    sum_local = 0.0_c_double
!$omp parallel if(LEN_1D > 1000)
!$omp do reduction(+:sum_local)
    do i = 1, LEN_1D
      sum_local = sum_local + a(i)
    end do
!$omp end do
!$omp end parallel
    sum_out(1) = sum_local
  end subroutine tsvc_2_s311_fp64

end module tsvc_2_s311_mod
