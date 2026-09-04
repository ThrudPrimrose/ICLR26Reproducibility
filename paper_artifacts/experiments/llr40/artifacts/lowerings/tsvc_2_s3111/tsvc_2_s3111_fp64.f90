! hpcagent_bench-autogen -- generated from tsvc_2_s3111_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3111_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(2)
    integer(c_int64_t) :: i_l0
    real(c_double) :: sum_val
    sum_val = 0.0_c_double
    do i_l0 = 0, (LEN_1D) - 1
        if ((a((i_l0) + 1) > 0.0_c_double)) then
            sum_val = (sum_val + a((i_l0) + 1))
        end if
    end do
    b((0) + 1) = sum_val

end subroutine tsvc_2_s3111_fp64
