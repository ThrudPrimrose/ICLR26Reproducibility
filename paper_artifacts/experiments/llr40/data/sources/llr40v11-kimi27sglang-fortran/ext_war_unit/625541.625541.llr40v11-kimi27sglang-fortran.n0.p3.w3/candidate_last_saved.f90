subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(c, name='ext_war_unit_fp64')
    use iso_c_binding, only: c_double, c_int64_t
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int64_t) :: i

    !$omp simd nontemporal(a)
    do i = 1, LEN_1D - 1
        a(i) = a(i + 1) + b(i)
    end do
    !$omp end simd
end subroutine ext_war_unit_fp64
