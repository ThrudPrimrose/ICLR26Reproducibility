subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_bytes) bind(C, name="scan_affine_decay_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: y(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    integer(c_intptr_t), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer(c_int64_t) :: i

    if (LEN_1D <= 1) return
    do i = 2, LEN_1D
        y(i) = c(i) * y(i-1) + x(i)
    end do
end subroutine scan_affine_decay_fp64
