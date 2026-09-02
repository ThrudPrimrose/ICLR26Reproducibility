subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: sum_out(1)
    integer(c_int64_t) :: i
    real(c_double) :: s

    s = 0.0_c_double
    do i = 1, LEN_1D
        s = s + a(i)
    end do
    sum_out(1) = s
end subroutine tsvc_2_s311_fp64
