! hpcagent_bench-autogen -- generated from tsvc_2_s2275_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s2275_fp32(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name="tsvc_2_s2275_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_float), intent(inout) :: a(LEN_2D)
    real(c_float), intent(inout) :: aa(LEN_2D, LEN_2D)
    real(c_float), intent(in) :: b(LEN_2D)
    real(c_float), intent(in) :: bb(LEN_2D, LEN_2D)
    real(c_float), intent(in) :: c(LEN_2D)
    real(c_float), intent(in) :: cc(LEN_2D, LEN_2D)
    real(c_float), intent(in) :: d(LEN_2D)
    integer(c_int64_t) :: i_l0, j_l1

    do i_l0 = 0, (LEN_2D) - 1
        do j_l1 = 0, (LEN_2D) - 1
            aa((i_l0) + 1, (j_l1) + 1) = (aa((i_l0) + 1, (j_l1) + 1) + (bb((i_l0) + 1, (j_l1) + 1) * cc((i_l0) + 1, &
            &(j_l1) + 1)))
        end do
        a((i_l0) + 1) = (b((i_l0) + 1) + (c((i_l0) + 1) * d((i_l0) + 1)))
    end do

end subroutine tsvc_2_s2275_fp32
