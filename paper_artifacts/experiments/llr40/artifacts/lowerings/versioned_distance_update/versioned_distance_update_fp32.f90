! hpcagent_bench-autogen -- generated from versioned_distance_update_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine versioned_distance_update_fp32(a, b, c, K, LEN_1D) bind(C, name="versioned_distance_update_fp32")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_float), intent(inout) :: a(LEN_1D)
    real(c_float), intent(in) :: b(LEN_1D)
    real(c_float), intent(in) :: c(LEN_1D)
    integer(c_int64_t) :: i_l0

    do i_l0 = K, (LEN_1D) - 1
        a((i_l0) + 1) = ((0.75_c_float * a(((i_l0 - K)) + 1)) + (b((i_l0) + 1) * c((i_l0) + 1)))
    end do

end subroutine versioned_distance_update_fp32
