! hpcagent_bench-autogen -- generated from tsvc_2_s318_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s318_fp32(a, result, LEN_1D, inc) bind(C, name="tsvc_2_s318_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: inc
    real(c_float), intent(in) :: a(LEN_1D)
    real(c_float), intent(inout) :: result(1)
    integer(c_int64_t) :: i_l0
    integer(c_int64_t) :: k
    real(c_float) :: index
    real(c_float) :: maxv
    real(c_float) :: v
    k = 0
    index = 0
    maxv = ABS(a((0) + 1))
    k = (k + inc)
    do i_l0 = 1, (LEN_1D) - 1
        v = ABS(a((k) + 1))
        if ((v > maxv)) then
            index = i_l0
            maxv = v
        end if
        k = (k + inc)
    end do
    result((0) + 1) = (maxv + index)

end subroutine tsvc_2_s318_fp32
