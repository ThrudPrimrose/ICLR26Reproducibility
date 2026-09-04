! hpcagent_bench-autogen -- generated from fuse_move_ifs_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine fuse_move_ifs_fp64(a, b, cond, src, K, LEN_2D) bind(C, name="fuse_move_ifs_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: b(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: cond(LEN_2D)
    real(c_double), intent(in) :: src(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i_l0, i_l2, j_l1, j_l3

    do i_l0 = 0, (LEN_2D) - 1
        if ((cond((i_l0) + 1) > 0.0_c_double)) then
            do j_l1 = 0, (LEN_2D) - 1
                a((j_l1) + 1, (i_l0) + 1) = (src((j_l1) + 1, (i_l0) + 1) * 2.0_c_double)
            end do
        end if
    end do
    if ((K > 0)) then
        do i_l2 = 0, (LEN_2D) - 1
            do j_l3 = 0, (LEN_2D) - 1
                b((j_l3) + 1, (i_l2) + 1) = (src((j_l3) + 1, (i_l2) + 1) + 1.0_c_double)
            end do
        end do
    end if

end subroutine fuse_move_ifs_fp64
