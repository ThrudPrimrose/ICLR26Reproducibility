! hpcagent_bench-autogen -- generated from compact_threshold_pack_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D) bind(C, &
&name="compact_threshold_pack_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), intent(inout) :: out_count(1)
    real(c_double), intent(inout) :: packed(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    real(c_double), intent(in) :: weight(LEN_1D)
    integer(c_int64_t) :: i_l0
    integer(c_int64_t) :: n
    n = 0
    do i_l0 = 0, (LEN_1D) - 1
        if ((src((i_l0) + 1) > 0.0_c_double)) then
            packed((n) + 1) = (src((i_l0) + 1) * weight((i_l0) + 1))
            n = (n + 1)
        end if
    end do
    out_count((0) + 1) = n

end subroutine compact_threshold_pack_fp64
