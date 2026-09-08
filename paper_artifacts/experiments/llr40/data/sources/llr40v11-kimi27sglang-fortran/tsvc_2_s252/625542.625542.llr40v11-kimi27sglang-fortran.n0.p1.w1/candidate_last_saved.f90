subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(c)
    use iso_c_binding, only: c_int64_t, c_double
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int64_t) :: i
    real(c_double) :: t, s

    if (LEN_1D < 64) then
        t = 0.0_c_double
        do i = 1, LEN_1D
            s = b(i) * c(i)
            a(i) = s + t
            t = s
        end do
    else
        a(1) = b(1) * c(1)
        !$omp simd safelen(8)
        do i = 2, LEN_1D
            a(i) = b(i) * c(i) + b(i-1) * c(i-1)
        end do
    end if
end subroutine tsvc_2_s252_fp64
