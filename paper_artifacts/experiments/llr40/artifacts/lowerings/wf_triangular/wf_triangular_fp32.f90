! hpcagent_bench-autogen -- generated from wf_triangular_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine wf_triangular_fp32(a, LEN_2D) bind(C, name="wf_triangular_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_float), intent(inout) :: a(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i_l0, j_l1

    do i_l0 = 1, (LEN_2D) - 1
        do j_l1 = i_l0, (LEN_2D) - 1
            a((j_l1) + 1, (i_l0) + 1) = ((a((j_l1) + 1, (i_l0) + 1) + a((j_l1) + 1, ((i_l0 - 1)) + 1)) + a(((j_l1 - &
            &1)) + 1, (i_l0) + 1))
        end do
    end do

end subroutine wf_triangular_fp32
