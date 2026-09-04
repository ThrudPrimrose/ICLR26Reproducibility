! hpcagent_bench-autogen -- generated from tsvc_2_s4112_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = 0, (LEN_1D) - 1
        a((i_l0) + 1) = (a((i_l0) + 1) + (b(INT(ip((i_l0) + 1), c_int64_t)) * 2.0_c_double))
    end do

end subroutine tsvc_2_s4112_fp64
