subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D, workspace, workspace_size) bind(c, name='tsvc_2_s233_fp64')
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(inout) :: bb(*)
    real(c_double), intent(in) :: cc(*)
    integer(c_int64_t), intent(in), value :: LEN_2D
    integer(c_int8_t), intent(in) :: workspace(*)
    integer(c_int64_t), intent(in), value :: workspace_size

    integer(c_int64_t) :: i, j

    do i = 8_c_int64_t, LEN_2D - 1_c_int64_t
        do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
            aa(j * LEN_2D + i + 1_c_int64_t) = aa((j - 1_c_int64_t) * LEN_2D + i + 1_c_int64_t) + cc(j * LEN_2D + i + 1_c_int64_t)
        end do
        do j = 8_c_int64_t, LEN_2D - 1_c_int64_t
            bb(j * LEN_2D + i + 1_c_int64_t) = bb(j * LEN_2D + (i - 1_c_int64_t) + 1_c_int64_t) + cc(j * LEN_2D + i + 1_c_int64_t)
        end do
    end do
end subroutine tsvc_2_s233_fp64
