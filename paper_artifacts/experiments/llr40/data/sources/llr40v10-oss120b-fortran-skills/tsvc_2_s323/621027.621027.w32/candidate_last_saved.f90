subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(LEN_1D)
    real(c_double), intent(in)    :: c(LEN_1D)
    real(c_double), intent(in)    :: d(LEN_1D)
    real(c_double), intent(in)    :: e(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: sum, cd, ce
    real(c_double) :: b1

    if (LEN_1D <= 1_c_int64_t) return

    b1 = b(1)
    !$omp parallel do default(none) shared(LEN_1D, a, b, c, d, e, b1) private(i, cd, ce)reduction(inscan, +: sum)
    do i = 2_c_int64_t, LEN_1D
        cd = c(i) * d(i)
        ce = c(i) * e(i)
        sum = sum + cd
        sum = sum + ce
        !$omp scan inclusive(sum)
        b(i) = b1 + sum
        a(i) = b1 + sum - ce
    end do
    !$omp end parallel do
end subroutine tsvc_2_s323_fp64
