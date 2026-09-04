! hpcagent_bench-autogen -- generated from tsvc_2_s2710_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(inout) :: c(LEN_1D)
    real(c_double), intent(in) :: d(LEN_1D)
    real(c_double), intent(in) :: e(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = 0, (LEN_1D) - 1
        if ((a((i_l0) + 1) > b((i_l0) + 1))) then
            a((i_l0) + 1) = (a((i_l0) + 1) + (b((i_l0) + 1) * d((i_l0) + 1)))
            if ((LEN_1D > 10)) then
                c((i_l0) + 1) = (c((i_l0) + 1) + (d((i_l0) + 1) * d((i_l0) + 1)))
            else
                c((i_l0) + 1) = ((d((i_l0) + 1) * e((i_l0) + 1)) + 1.0_c_double)
            end if
        else
            b((i_l0) + 1) = (a((i_l0) + 1) + (e((i_l0) + 1) * e((i_l0) + 1)))
            if ((x((0) + 1) > 0.0_c_double)) then
                c((i_l0) + 1) = (a((i_l0) + 1) + (d((i_l0) + 1) * d((i_l0) + 1)))
            else
                c((i_l0) + 1) = (c((i_l0) + 1) + (e((i_l0) + 1) * e((i_l0) + 1)))
            end if
        end if
    end do

end subroutine tsvc_2_s2710_fp64
