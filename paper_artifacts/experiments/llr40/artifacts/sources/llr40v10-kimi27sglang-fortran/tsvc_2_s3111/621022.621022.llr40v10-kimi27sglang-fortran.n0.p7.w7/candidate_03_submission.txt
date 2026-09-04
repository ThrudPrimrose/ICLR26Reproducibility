subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3111_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(2)

    integer(c_int64_t) :: i
    real(c_double) :: sum_val

    sum_val = 0.0_c_double
    !$omp parallel do simd reduction(+:sum_val) schedule(static)
    do i = 1, LEN_1D
        if (a(i) > 0.0_c_double) then
            sum_val = sum_val + a(i)
        end if
    end do
    !$omp end parallel do simd
    b(1) = sum_val
end subroutine tsvc_2_s3111_fp64
