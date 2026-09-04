! hpcagent_bench-autogen -- generated from ext_break_capture_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), parameter :: K = 1_8
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    integer(c_int64_t), intent(inout) :: out_index(1)
    real(c_double), intent(inout) :: out_value(1)
    integer(c_int64_t) :: i_l0

    out_index((0) + 1) = ((-1)) + 1
    out_value((0) + 1) = (-(1.0_c_double))
    do i_l0 = 0, (LEN_1D) - 1
        if ((a((i_l0) + 1) > K)) then
            out_index((0) + 1) = (i_l0) + 1
            out_value((0) + 1) = a((i_l0) + 1)
            exit
        end if
    end do

end subroutine ext_break_capture_fp64
