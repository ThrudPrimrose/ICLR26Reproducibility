subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(c, name='tsvc_2_s2710_fp64')
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), dimension(LEN_1D), intent(inout) :: a, b, c
    real(c_double), dimension(LEN_1D), intent(in) :: d, e, x
    integer(c_int64_t) :: i
    logical :: len_gt, x_pos
    real(c_double) :: ai, bi, ci, di, ei
    real(c_double) :: a_new, b_new, c_gt, c_le
    real(c_double) :: one

    one = 1.0_c_double
    len_gt = LEN_1D > 10_c_int64_t
    x_pos = x(1) > 0.0_c_double

    !$omp parallel do simd
    do i = 1, LEN_1D
        ai = a(i)
        bi = b(i)
        ci = c(i)
        di = d(i)
        ei = e(i)

        a_new = ai + bi * di
        b_new = ai + ei * ei

        c_gt = merge(ci + di * di, di * ei + one, len_gt)
        c_le = merge(ai + di * di, ci + ei * ei, x_pos)

        a(i) = merge(a_new, ai, ai > bi)
        b(i) = merge(bi, b_new, ai > bi)
        c(i) = merge(c_gt, c_le, ai > bi)
    end do
    !$omp end parallel do simd
end subroutine tsvc_2_s2710_fp64
