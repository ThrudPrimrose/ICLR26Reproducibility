subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_size) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: y(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: blk, nb, block_size, n_elem
    integer(c_int64_t) :: blk_start, blk_end, idx
    real(c_double) :: A_loc, B_loc, cur
    real(c_double), allocatable :: block_A(:), block_B(:), block_seed(:)

    if (LEN_1D <= 1_c_int64_t) return
    n_elem = LEN_1D - 1_c_int64_t
    block_size = 4096_c_int64_t
    nb = (n_elem + block_size - 1_c_int64_t) / block_size
    allocate(block_A(nb), block_B(nb), block_seed(nb))

    ! Phase 1: compute block transformation (A, B) for each block
    !$omp parallel do default(none) shared(c, x, block_A, block_B, nb, block_size, n_elem, LEN_1D) &
!$omp& private(blk, blk_start, blk_end, idx, A_loc, B_loc) schedule(static)
    do blk = 1, nb
        blk_start = (blk - 1) * block_size + 2_c_int64_t
        blk_end = min(LEN_1D, blk_start + block_size - 1_c_int64_t)
        A_loc = 1.0_c_double
        B_loc = 0.0_c_double
        do idx = blk_start, blk_end
            A_loc = c(idx) * A_loc
            B_loc = c(idx) * B_loc + x(idx)
        end do
        block_A(blk) = A_loc
        block_B(blk) = B_loc
    end do
    !$omp end parallel do

    ! Phase 2: compute seed (starting y) for each block
    block_seed(1) = y(1)
    do blk = 2, nb
        block_seed(blk) = block_A(blk-1) * block_seed(blk-1) + block_B(blk-1)
    end do

    ! Phase 3: compute y values in each block using its seed
    !$omp parallel do default(none) shared(c, x, y, block_seed, nb, block_size, LEN_1D) &
!$omp& private(blk, blk_start, blk_end, idx, cur) schedule(static)
    do blk = 1, nb
        blk_start = (blk - 1) * block_size + 2_c_int64_t
        blk_end = min(LEN_1D, blk_start + block_size - 1_c_int64_t)
        cur = block_seed(blk)
        do idx = blk_start, blk_end
            cur = c(idx) * cur + x(idx)
            y(idx) = cur
        end do
    end do
    !$omp end parallel do

    deallocate(block_A, block_B, block_seed)
end subroutine scan_affine_decay_fp64
