! hpcagent_bench-autogen -- generated from fuse_diamond_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name="fuse_diamond_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(LEN_1D)
    integer(c_int64_t) :: i_l0, i_l1, i_l2, i_l3
    real(c_double) :: t(LEN_1D)
    real(c_double) :: u(LEN_1D)
    real(c_double) :: v(LEN_1D)
    do i_l0 = 0, (LEN_1D) - 1
        t((i_l0) + 1) = (a((i_l0) + 1) * a((i_l0) + 1))
    end do
    do i_l1 = 0, (LEN_1D) - 1
        u((i_l1) + 1) = (t((i_l1) + 1) + 1.0_c_double)
    end do
    do i_l2 = 0, (LEN_1D) - 1
        v((i_l2) + 1) = (t((i_l2) + 1) - 1.0_c_double)
    end do
    do i_l3 = 0, (LEN_1D) - 1
        out((i_l3) + 1) = (u((i_l3) + 1) * v((i_l3) + 1))
    end do

end subroutine fuse_diamond_fp64
