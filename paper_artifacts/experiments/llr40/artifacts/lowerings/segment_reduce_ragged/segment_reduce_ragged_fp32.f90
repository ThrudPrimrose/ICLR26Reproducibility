! hpcagent_bench-autogen -- generated from segment_reduce_ragged_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine segment_reduce_ragged_fp32(out, row_ptr, val, w, NSEG) bind(C, name="segment_reduce_ragged_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: NSEG
    real(c_float), intent(inout) :: out(NSEG)
    integer(c_int64_t), intent(in) :: row_ptr((NSEG + 1))
    real(c_float), intent(in) :: val((NSEG * 24))
    real(c_float), intent(in) :: w((NSEG * 24))
    integer(c_int64_t) :: e_l1, s_l0
    real(c_float) :: acc
    do s_l0 = 0, (NSEG) - 1
        acc = 0.0_c_float
        do e_l1 = row_ptr((s_l0) + 1), (row_ptr(((s_l0 + 1)) + 1)) - 1
            acc = (acc + (val((e_l1) + 1) * w((e_l1) + 1)))
        end do
        out((s_l0) + 1) = acc
    end do

end subroutine segment_reduce_ragged_fp32
