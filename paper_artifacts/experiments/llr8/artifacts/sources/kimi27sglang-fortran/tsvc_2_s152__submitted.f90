subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(c, name="tsvc_2_s152_fp64")
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*), b(*)
    real(c_double), intent(in)    :: c(*), d(*), e(*)
    integer(c_int64_t), value     :: LEN_1D
    integer(c_int64_t)            :: i

    !$omp parallel do simd
    do i = 1, LEN_1D
        b(i) = d(i) * e(i)
        a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do simd
end subroutine tsvc_2_s152_fp64
