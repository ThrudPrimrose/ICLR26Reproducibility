! hpcagent_bench-autogen -- generated from tsvc_2_s3112_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s3112_fp32(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_float), intent(in) :: a(LEN_1D)
    real(c_float), intent(inout) :: b(LEN_1D)
    integer(c_int64_t) :: i_l0
    real(c_float) :: sum
    sum = 0.0_c_float
    do i_l0 = 0, (LEN_1D) - 1
        sum = (sum + a((i_l0) + 1))
        b((i_l0) + 1) = sum
    end do

end subroutine tsvc_2_s3112_fp32
