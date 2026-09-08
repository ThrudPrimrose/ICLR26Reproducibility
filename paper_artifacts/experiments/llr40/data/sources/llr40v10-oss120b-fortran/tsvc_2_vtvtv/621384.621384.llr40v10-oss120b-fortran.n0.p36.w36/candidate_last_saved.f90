subroutine tsvc_2_vtvtv_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_vtvtv_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), dimension(*), intent(inout) :: a
    real(c_double), dimension(*), intent(in) :: b
    real(c_double), dimension(*), intent(in) :: c
    integer(c_int64_t) :: i

    !$omp parallel do default(none) shared(a,b,c,LEN_1D) private(i) schedule(static)
    do i = 1, LEN_1D
        a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do
end subroutine tsvc_2_vtvtv_fp64
