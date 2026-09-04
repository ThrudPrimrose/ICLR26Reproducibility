! hpcagent_bench-autogen -- generated from quasi_affine_reduce_odd_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine quasi_affine_reduce_odd_fp32(a, out, LEN_1D) bind(C, name="quasi_affine_reduce_odd_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_float), intent(in) :: a(LEN_1D)
    real(c_float), intent(inout) :: out(1)
    integer(c_int64_t) :: i_l0

    out((0) + 1) = 0.0_c_float
    do i_l0 = 1, (LEN_1D) - 1, 2
        out((0) + 1) = (out((0) + 1) + a((i_l0) + 1))
    end do

end subroutine quasi_affine_reduce_odd_fp32
