subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc, workspace, workspace_size) bind(C, name="tsvc_2_s318_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D, inc
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    real(c_double) :: maxv, v
    integer(c_int64_t) :: i, k, idx0
    ! Handle empty case (should not happen)
    if (LEN_1D <= 0_c_int64_t) then
        result(1) = 0.0_c_double
        return
    end if
    ! Compute maximum absolute value using parallel reduction
    maxv = 0.0_c_double
    !$omp parallel do private(i, k, v) reduction(max:maxv) schedule(static)
    do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
        k = 1_c_int64_t + i * inc
        v = abs(a(k))
        if (v > maxv) maxv = v
    end do
    !$omp end parallel do
    ! Find the first index where the maximum occurs (0-based)
    idx0 = -1_c_int64_t
    do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
        k = 1_c_int64_t + i * inc
        if (abs(a(k)) == maxv) then
            idx0 = i
            exit
        end if
    end do
    result(1) = maxv + real(idx0, kind=c_double)
end subroutine tsvc_2_s318_fp64
