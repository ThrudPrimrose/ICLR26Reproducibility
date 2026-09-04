! hpcagent_bench-autogen -- generated from tsvc_2_s255_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s255_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i_l0
    real(c_double) :: x
    real(c_double) :: y
    x = b(((LEN_1D - 1)) + 1)
    y = b(((LEN_1D - 2)) + 1)
    do i_l0 = 0, (LEN_1D) - 1
        a((i_l0) + 1) = (((b((i_l0) + 1) + x) + y) * 0.333_c_double)
        y = x
        x = b((i_l0) + 1)
    end do

end subroutine tsvc_2_s255_fp64
