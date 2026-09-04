! hpcagent_bench-autogen -- generated from argmax_with_index_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine argmax_with_index_fp32(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_float), intent(in) :: a(LEN_1D)
    integer(c_int64_t), intent(inout) :: out_index(1)
    real(c_float), intent(inout) :: out_value(1)
    integer(c_int64_t) :: i_l0
    real(c_float) :: x
    real(c_float) :: idx
    x = a((0) + 1)
    idx = 0
    do i_l0 = 1, (LEN_1D) - 1
        if ((a((i_l0) + 1) > x)) then
            x = a((i_l0) + 1)
            idx = i_l0
        end if
    end do
    out_value((0) + 1) = x
    out_index((0) + 1) = (idx) + 1

end subroutine argmax_with_index_fp32
