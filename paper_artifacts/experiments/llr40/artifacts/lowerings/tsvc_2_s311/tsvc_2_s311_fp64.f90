! hpcagent_bench-autogen -- generated from tsvc_2_s311_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C, name="tsvc_2_s311_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: sum_out(LEN_1D)
    integer(c_int64_t) :: i_l0

    sum_out((0) + 1) = 0.0_c_double
    do i_l0 = 0, (LEN_1D) - 1
        sum_out((0) + 1) = (sum_out((0) + 1) + a((i_l0) + 1))
    end do

end subroutine tsvc_2_s311_fp64
