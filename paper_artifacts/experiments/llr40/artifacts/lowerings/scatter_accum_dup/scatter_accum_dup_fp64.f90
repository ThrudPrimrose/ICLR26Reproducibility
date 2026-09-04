! hpcagent_bench-autogen -- generated from scatter_accum_dup_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: bins(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = 0, (LEN_1D) - 1
        bins(INT(ip((i_l0) + 1), c_int64_t)) = (bins(INT(ip((i_l0) + 1), c_int64_t)) + src((i_l0) + 1))
    end do

end subroutine scatter_accum_dup_fp64
