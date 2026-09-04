!scan_affine_decay.f90
! Parallel scan implementation for variable coefficient recurrence.
subroutine scan_affine_decay_fp64(LEN_1D, y, c, x, workspace, workspace_size) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: y(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size

    integer(c_int64_t), parameter :: BLOCK_SIZE = 256_c_int64_t
    integer(c_int64_t) :: N, num_blocks, b, i, start_idx, end_idx
    real(c_double) :: a_loc, b_loc, y_val
    real(c_double), allocatable :: block_a(:), block_b(:)
    real(c_double), allocatable :: prefix_a(:), prefix_b(:)

    N = LEN_1D
    print *, 'DEBUG N =', N
    if (N <= 1) return
    ! Ensure seed is set (y(1) = x(1))
    y(1) = x(1)

    num_blocks = (N - 1 + BLOCK_SIZE - 1) / BLOCK_SIZE
    if (num_blocks < 1) num_blocks = 1

    allocate(block_a(num_blocks), block_b(num_blocks))
    allocate(prefix_a(num_blocks+1), prefix_b(num_blocks+1))

    !$omp parallel do private(b, i, start_idx, end_idx, a_loc, b_loc) schedule(static)
    do b = 1, num_blocks
        a_loc = 1.0d0
        b_loc = 0.0d0
        start_idx = 2 + (b-1) * BLOCK_SIZE
        end_idx = min(N, 2 + b * BLOCK_SIZE - 1)
        if (start_idx <= end_idx) then
            do i = start_idx, end_idx
                a_loc = c(i) * a_loc
                b_loc = c(i) * b_loc + x(i)
            end do
        end if
        block_a(b) = a_loc
        block_b(b) = b_loc
    end do
    !$omp end parallel do

    ! Prefix over blocks (serial)
    prefix_a(1) = 1.0d0
    prefix_b(1) = 0.0d0
    do b = 1, num_blocks
        prefix_a(b+1) = block_a(b) * prefix_a(b)
        prefix_b(b+1) = block_a(b) * prefix_b(b) + block_b(b)
    end do

    ! Apply transformations to compute final y in parallel
    !$omp parallel do private(b, i, start_idx, end_idx, y_val) schedule(static)
    do b = 1, num_blocks
        y_val = prefix_a(b) * y(1) + prefix_b(b)
        start_idx = 2 + (b-1) * BLOCK_SIZE
        end_idx = min(N, 2 + b * BLOCK_SIZE - 1)
        if (start_idx <= end_idx) then
            do i = start_idx, end_idx
                y(i) = c(i) * y_val + x(i)
                y_val = y(i)
            end do
        end if
    end do
    !$omp end parallel do

    deallocate(block_a, block_b, prefix_a, prefix_b)

end subroutine scan_affine_decay_fp64
