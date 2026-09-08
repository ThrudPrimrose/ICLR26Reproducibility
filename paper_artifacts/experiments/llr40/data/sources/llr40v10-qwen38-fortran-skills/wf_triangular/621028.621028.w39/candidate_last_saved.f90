subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j
    do i = 2, LEN_2D
        do j = i, LEN_2D
            a(j, i) = a(j, i) + a(j, i-1) + a(j-1, i)
        end do
    end do
end subroutine wf_triangular_fp64
