subroutine wf_triangular_fp64(a, LEN_2D, workspace, workspace_bytes) bind(C, name="wf_triangular_fp64")
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*)
    integer(c_int64_t), value :: LEN_2D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer :: i, j, k
    integer :: i_start, i_end
    integer :: idx
    integer :: len2
    ! Convert LEN_2D (C int64) to default integer for indexing.
    len2 = LEN_2D
    ! Wavefront parallelism using row-major indexing on the flattened array.
    !$omp parallel private(i, j, k, i_start, i_end, idx)
    do k = 4, 2 * len2
        i_start = max(2, k - len2)
        i_end   = min(len2, k / 2)
        !$omp do schedule(static)
        do i = i_start, i_end
            j = k - i
            idx = (i - 1) * len2 + j
            a(idx) = a(idx) + a(idx - len2) + a(idx - 1)
        end do
        !$omp end do
    end do
    !$omp end parallel
end subroutine wf_triangular_fp64
