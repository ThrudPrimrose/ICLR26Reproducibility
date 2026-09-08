subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C)
    use iso_c_binding
    implicit none
    ! Arguments
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: result(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: inc
    ! Local variables
    integer(c_int64_t) :: i, k
    real(c_double) :: v
    real(c_double) :: maxv
    integer(c_int64_t) :: idx
    ! Per-thread temporaries for reduction
    real(c_double) :: local_maxv
    integer(c_int64_t) :: local_idx
    ! Initialize global maximum to -infinity and index to -1
    maxv = -huge(0.0_c_double)
    idx = -1_c_int64_t
    !$omp parallel private(i, k, v, local_maxv, local_idx) shared(maxv, idx)
        local_maxv = -huge(0.0_c_double)
        local_idx = -1_c_int64_t
        !$omp do schedule(static)
        do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
            k = i * inc
            v = abs(a(k+1))
            if (v > local_maxv) then
                local_maxv = v
                local_idx = i
            end if
        end do
        !$omp end do
        !$omp critical
            if (local_maxv > maxv) then
                maxv = local_maxv
                idx = local_idx
            else if (local_maxv == maxv) then
                if (local_idx < idx) then
                    idx = local_idx
                end if
            end if
        !$omp end critical
    !$omp end parallel
    result(1) = maxv + real(idx, kind=c_double)
end subroutine tsvc_2_s318_fp64
