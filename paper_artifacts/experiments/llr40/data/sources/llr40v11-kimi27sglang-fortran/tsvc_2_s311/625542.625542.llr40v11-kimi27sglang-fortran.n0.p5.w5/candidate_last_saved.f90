module tsvc_2_s311_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name='tsvc_2_s311_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: sum_out
    real(c_double) :: s
    integer(c_int64_t) :: i

    s = 0.0_c_double
    !$omp parallel do simd reduction(+:s) schedule(static)
    do i = 1, LEN_1D
      s = s + a(i)
    end do
    !$omp end parallel do simd
    sum_out = s
  end subroutine tsvc_2_s311_fp64
end module tsvc_2_s311_mod
