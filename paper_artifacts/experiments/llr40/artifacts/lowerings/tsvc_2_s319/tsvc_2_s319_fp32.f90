! hpcagent_bench-autogen -- generated from tsvc_2_s319_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s319_fp32(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s319_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_float), intent(inout) :: a(LEN_1D)
    real(c_float), intent(inout) :: b(LEN_1D)
    real(c_float), intent(in) :: c(LEN_1D)
    real(c_float), intent(in) :: d(LEN_1D)
    real(c_float), intent(in) :: e(LEN_1D)
    integer(c_int64_t) :: i_l0
    real(c_float) :: sum_val
    sum_val = 0.0_c_float
    do i_l0 = 0, (LEN_1D) - 1
        a((i_l0) + 1) = (c((i_l0) + 1) + d((i_l0) + 1))
        sum_val = (sum_val + a((i_l0) + 1))
        b((i_l0) + 1) = (c((i_l0) + 1) + e((i_l0) + 1))
        sum_val = (sum_val + b((i_l0) + 1))
    end do
    b((0) + 1) = sum_val

end subroutine tsvc_2_s319_fp32
