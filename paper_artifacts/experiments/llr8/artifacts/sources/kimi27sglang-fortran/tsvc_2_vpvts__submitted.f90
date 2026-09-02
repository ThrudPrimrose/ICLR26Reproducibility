subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(c, name='tsvc_2_vpvts_fp64')
    use iso_c_binding, only: c_int64_t, c_double
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D, S
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: ds

    ds = real(S, c_double)
    !$omp parallel do simd schedule(static) if(len_1d > 1000)
    do i = 1, LEN_1D
        a(i) = a(i) + b(i) * ds
    end do
    !$omp end parallel do simd
end subroutine tsvc_2_vpvts_fp64
