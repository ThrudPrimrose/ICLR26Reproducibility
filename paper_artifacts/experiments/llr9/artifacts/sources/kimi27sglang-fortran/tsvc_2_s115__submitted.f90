subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(c, name="tsvc_2_s115_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: aa(*)
    integer(c_int64_t) :: i, j

    do j = 1, LEN_2D
        !$omp simd
        do i = j + 1, LEN_2D
            a(i) = a(i) - aa(i + (j - 1) * LEN_2D) * a(j)
        end do
    end do
end subroutine tsvc_2_s115_fp64
