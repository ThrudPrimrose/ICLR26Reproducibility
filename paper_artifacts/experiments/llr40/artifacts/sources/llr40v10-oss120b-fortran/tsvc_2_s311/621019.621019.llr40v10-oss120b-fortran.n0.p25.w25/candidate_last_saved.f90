! Fortran implementation of the TSVC tsvc_2 kernel s311 (double precision sum).
!
! The kernel computes the sum of an array 'a' of length LEN_1D and stores the
! result in sum_out(1). The interface follows the C-ABI used by the benchmark:
!   void tsvc_2_s311_fp64(const double *restrict a,
!                         double *restrict sum_out,
!                         const int64_t LEN_1D);
!
! The Fortran subroutine uses BIND(C) to expose the same symbol name and calls
! OpenMP to parallelize the reduction across threads. The compiler is invoked with
! -O3 -march=native -fopenmp, which enables auto-vectorisation of the inner loop.
!
! Author: ChatGPT
!
module tsvc_2_s311_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains

  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    ! Arguments as defined by the C reference.
    real(c_double), intent(in)  :: a(*)          ! Input array
    real(c_double), intent(inout) :: sum_out(*) ! Output scalar stored at index 1
    integer(c_int64_t), value   :: LEN_1D       ! Length, passed by value
    integer(c_int64_t)          :: i
    real(c_double)              :: local_sum

    local_sum = 0.0_c_double
    !$omp parallel do reduction(+:local_sum) schedule(static)
    do i = 1, LEN_1D
      local_sum = local_sum + a(i)
    end do
    !$omp end parallel do

    sum_out(1) = local_sum
  end subroutine tsvc_2_s311_fp64

end module tsvc_2_s311_mod
