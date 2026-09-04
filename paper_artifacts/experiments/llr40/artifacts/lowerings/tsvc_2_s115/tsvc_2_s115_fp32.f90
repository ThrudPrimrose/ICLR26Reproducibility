! hpcagent_bench-autogen -- generated from tsvc_2_s115_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s115_fp32(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_float), intent(inout) :: a(LEN_2D)
    real(c_float), intent(in) :: aa(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i_l1, j_l0

    do j_l0 = 0, (LEN_2D) - 1
        do i_l1 = (j_l0 + 1), (LEN_2D) - 1
            a((i_l1) + 1) = (a((i_l1) + 1) - (aa((i_l1) + 1, (j_l0) + 1) * a((j_l0) + 1)))
        end do
    end do

end subroutine tsvc_2_s115_fp32
