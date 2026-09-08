subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(C, name="tsvc_2_s255_fp64")
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t), value, intent(in) :: len_1d
    integer(c_int64_t) :: i
    real(c_double), parameter :: factor = 0.333d0
    !$omp parallel do schedule(static)
    do i = 1, len_1d
        a(i) = (b(i) + b(modulo(i-2, len_1d) + 1) + b(modulo(i-3, len_1d) + 1)) * factor
    end do
    end subroutine tsvc_2_s255_fp64
