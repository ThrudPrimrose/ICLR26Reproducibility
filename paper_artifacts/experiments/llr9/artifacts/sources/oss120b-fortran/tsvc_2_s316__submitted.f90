subroutine tsvc_2_s316_fp64(a, result, LEN_1D, workspace, workspace_bytes) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(1)
    integer(c_int64_t) :: i
    real(c_double) :: x

    if (LEN_1D <= 0_c_int64_t) then
        result(1) = 0.0_c_double
        return
    end if

    x = a(1)

    !$omp parallel do reduction(min:x) private(i) schedule(static)
    do i = 2, LEN_1D
        if (a(i) < x) x = a(i)
    end do
    !$omp end parallel do

    result(1) = x
end subroutine tsvc_2_s316_fp64
