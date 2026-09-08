subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: sum_out(LEN_1D)
    real(c_double) :: sum
    integer(c_int64_t) :: i

    sum = 0.0_c_double
    !$omp parallel do reduction(+:sum)
    do i = 1, LEN_1D
        sum = sum + a(i)
    end do
    sum_out(1) = sum
end subroutine tsvc_2_s311_fp64
