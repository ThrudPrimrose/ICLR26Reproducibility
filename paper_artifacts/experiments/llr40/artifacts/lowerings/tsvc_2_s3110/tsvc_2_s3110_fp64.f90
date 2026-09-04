! hpcagent_bench-autogen -- generated from tsvc_2_s3110_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
    real(c_double), intent(inout) :: bb(2, 2)
    integer(c_int64_t) :: i_l0, j_l1
    real(c_double) :: maxv
    real(c_double) :: xindex
    real(c_double) :: yindex
    real(c_double) :: chksum
    real(c_double) :: tmp1
    real(c_double) :: tmp2
    maxv = aa((0) + 1, (0) + 1)
    xindex = 0
    yindex = 0
    do i_l0 = 0, (LEN_2D) - 1
        do j_l1 = 0, (LEN_2D) - 1
            if ((aa((j_l1) + 1, (i_l0) + 1) > maxv)) then
                maxv = aa((j_l1) + 1, (i_l0) + 1)
                xindex = i_l0
                yindex = j_l1
            end if
        end do
    end do
    chksum = ((maxv + xindex) + yindex)
    tmp1 = chksum
    tmp2 = tmp1
    bb((0) + 1, (0) + 1) = chksum

end subroutine tsvc_2_s3110_fp64
