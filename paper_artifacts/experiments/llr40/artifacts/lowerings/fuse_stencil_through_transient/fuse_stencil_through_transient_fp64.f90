! hpcagent_bench-autogen -- generated from fuse_stencil_through_transient_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D) bind(C, name="fuse_stencil_through_transient_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(LEN_1D)
    integer(c_int64_t) :: i_l0, i_l1
    real(c_double) :: tmp(LEN_1D)
    do i_l0 = 1, ((LEN_1D - 1)) - 1
        tmp((i_l0) + 1) = ((a(((i_l0 - 1)) + 1) + a((i_l0) + 1)) + a(((i_l0 + 1)) + 1))
    end do
    do i_l1 = 1, ((LEN_1D - 2)) - 1
        out((i_l1) + 1) = (tmp((i_l1) + 1) * tmp(((i_l1 + 1)) + 1))
    end do

end subroutine fuse_stencil_through_transient_fp64
