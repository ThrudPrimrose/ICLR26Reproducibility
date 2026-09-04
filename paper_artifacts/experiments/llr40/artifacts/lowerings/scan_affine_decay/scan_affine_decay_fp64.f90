! hpcagent_bench-autogen -- generated from scan_affine_decay_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine scan_affine_decay_fp64(c, x, y, LEN_1D) bind(C, name="scan_affine_decay_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    real(c_double), intent(inout) :: y(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = 1, (LEN_1D) - 1
        y((i_l0) + 1) = ((c((i_l0) + 1) * y(((i_l0 - 1)) + 1)) + x((i_l0) + 1))
    end do

end subroutine scan_affine_decay_fp64
