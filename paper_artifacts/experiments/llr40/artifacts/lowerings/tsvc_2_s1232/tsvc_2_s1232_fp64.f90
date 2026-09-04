! hpcagent_bench-autogen -- generated from tsvc_2_s1232_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN) bind(C, name="tsvc_2_s1232_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t), value, intent(in) :: VLEN
    real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i_l1, j_l0

    do j_l0 = 0, (LEN_2D) - 1
        do i_l1 = (j_l0 * VLEN), (LEN_2D) - 1
            aa((j_l0) + 1, (i_l1) + 1) = (bb((j_l0) + 1, (i_l1) + 1) + cc((j_l0) + 1, (i_l1) + 1))
        end do
    end do

end subroutine tsvc_2_s1232_fp64
