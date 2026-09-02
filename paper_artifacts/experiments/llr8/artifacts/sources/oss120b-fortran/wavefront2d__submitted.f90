subroutine wavefront2d_fp64(a, LEN_2D, workspace, workspace_bytes) bind(C, name="wavefront2d_fp64")
    use iso_c_binding, only: c_ptr, c_int64_t, c_double, c_f_pointer
    implicit none
    type(c_ptr), value :: a
    integer(c_int64_t), value :: LEN_2D
    type(c_ptr), value :: workspace      ! unused, kept for API compatibility
    integer(c_int64_t), value :: workspace_bytes
    real(c_double), pointer :: arr(:)
    integer(c_int64_t) :: i, j, k, i_start, i_end
    integer(c_int64_t) :: idx, idx_up, idx_left, idx_diag
    
    ! Associate the C pointer with a Fortran array view (row‑major layout)
    call c_f_pointer(a, arr, [LEN_2D*LEN_2D])
    !$omp parallel default(none) shared(arr, LEN_2D) private(i, j, k, i_start, i_end, idx, idx_up, idx_left, idx_diag)
    do k = 4_c_int64_t, 2_c_int64_t * LEN_2D
        i_start = max(2_c_int64_t, k - LEN_2D)
        i_end   = min(LEN_2D, k - 2_c_int64_t)
        !$omp do schedule(static)
        do i = i_start, i_end
            j = k - i
            idx      = (i - 1_c_int64_t) * LEN_2D + (j - 1_c_int64_t)
            idx_up   = (i - 2_c_int64_t) * LEN_2D + (j - 1_c_int64_t)
            idx_left = (i - 1_c_int64_t) * LEN_2D + (j - 2_c_int64_t)
            idx_diag = (i - 2_c_int64_t) * LEN_2D + (j - 2_c_int64_t)
            arr(idx + 1) = 0.25d0 * (arr(idx + 1) + arr(idx_up + 1) + arr(idx_left + 1) + arr(idx_diag + 1))
        end do
        !$omp end do
    end do
    !$omp end parallel
end subroutine wavefront2d_fp64
