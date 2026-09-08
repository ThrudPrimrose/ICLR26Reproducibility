subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double) :: a_orig(LEN_1D)
    integer(c_int64_t) :: i

    if (LEN_1D > 0) then
        !$omp parallel do simd default(none) shared(a,a_orig,LEN_1D) private(i)
        do i = 1, LEN_1D
            a_orig(i) = a(i)
        end do
    end if

    if (LEN_1D > 1) then
        !$omp parallel do simd default(none) shared(a,b,a_orig,LEN_1D) private(i)
        do i = 1, LEN_1D-1
            a(i) = a_orig(i+1) + b(i)
        end do
    end if

end subroutine ext_war_unit_fp64
