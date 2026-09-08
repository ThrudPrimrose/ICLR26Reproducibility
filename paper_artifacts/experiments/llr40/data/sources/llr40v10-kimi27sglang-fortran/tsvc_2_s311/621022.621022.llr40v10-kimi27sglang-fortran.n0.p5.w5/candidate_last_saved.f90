module tsvc_2_s311
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
contains
  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D, workspace, workspace_size) bind(c, name='tsvc_2_s311_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: a(0:LEN_1D-1)
    real(c_double), intent(inout) :: sum_out(0:LEN_1D-1)
    integer(c_int8_t), intent(inout) :: workspace(*)
    real(c_double) :: s
    integer(c_int64_t) :: i

    s = 0.0_c_double
    !$omp simd reduction(+:s)
    do i = 0, LEN_1D - 1
      s = s + a(i)
    end do
    !$omp end simd

    sum_out(0) = s
  end subroutine tsvc_2_s311_fp64
end module tsvc_2_s311
