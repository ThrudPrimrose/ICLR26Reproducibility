subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(c, name='tsvc_2_s2710_fp64')
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), dimension(LEN_1D), intent(inout) :: a, b, c
    real(c_double), dimension(LEN_1D), intent(in) :: d, e, x
    integer(c_int64_t) :: i
    real(c_double) :: mask_len, mask_x, mask_gt
    real(c_double) :: ai, bi, ci, di, ei, di2, ei2
    real(c_double) :: a_new, b_new, c_gt, c_le
    real(c_double) :: one

    one = 1.0_c_double
    mask_len = merge(one, 0.0_c_double, LEN_1D > 10_c_int64_t)
    mask_x = merge(one, 0.0_c_double, x(1) > 0.0_c_double)

    !$omp parallel do simd
    do i = 1, LEN_1D
        ai = a(i)
        bi = b(i)
        ci = c(i)
        di = d(i)
        ei = e(i)

        di2 = di * di
        ei2 = ei * ei
        mask_gt = merge(one, 0.0_c_double, ai > bi)

        a_new = ai + bi * di
        b_new = ai + ei2

        c_gt = mask_len * (ci + di2) + (one - mask_len) * (di * ei + one)
        c_le = mask_x * (ai + di2) + (one - mask_x) * (ci + ei2)

        a(i) = mask_gt * a_new + (one - mask_gt) * ai
        b(i) = mask_gt * bi + (one - mask_gt) * b_new
        c(i) = mask_gt * c_gt + (one - mask_gt) * c_le
    end do
    !$omp end parallel do simd
end subroutine tsvc_2_s2710_fp64
