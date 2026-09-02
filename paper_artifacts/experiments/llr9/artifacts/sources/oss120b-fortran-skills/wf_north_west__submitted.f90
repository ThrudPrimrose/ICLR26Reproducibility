! Wavefront north-west kernel (wf_north_west_fp64) - tiled wavefront for parallelism and exact numeric match
subroutine wf_north_west_fp64(a, n, workspace, workspace_size) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: n
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(n, n)
    integer(c_int64_t) :: B, nt, d, ti, tj, i0, i1, j0, j1, i, j
    ! Tile size chosen to balance cache usage and parallelism
    B = 64_c_int64_t
    nt = (n + B - 1) / B
    ! Loop over tile-diagonals (anti-diagonals of tiles)
    do d = 0_c_int64_t, 2 * (nt - 1_c_int64_t)
        !$omp parallel do private(ti, tj, i0, i1, j0, j1, i, j) schedule(static)
        do ti = max(0_c_int64_t, d - (nt - 1_c_int64_t)), min(nt - 1_c_int64_t, d)
            tj = d - ti
            ! Compute tile bounds in array indices (1‑based)
            i0 = ti * B + 1_c_int64_t
            i1 = min((ti + 1_c_int64_t) * B, n)
            j0 = tj * B + 1_c_int64_t
            j1 = min((tj + 1_c_int64_t) * B, n)
            ! Enforce the kernel's lower bounds (i,j >= 2)
            i0 = max(i0, 2_c_int64_t)
            j0 = max(j0, 2_c_int64_t)
            ! Process each tile in the original lexicographic order (i outer, j inner)
            do i = i0, i1
                !$omp simd
                do j = j0, j1
                    a(j, i) = a(j, i) + a(j - 1, i) + a(j, i - 1)
                end do
                !$omp end simd
            end do
        end do
        !$omp end parallel do
    end do
end subroutine wf_north_west_fp64
