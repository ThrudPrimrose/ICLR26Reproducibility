subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_size) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: y(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    real(c_double), intent(in) :: x(LEN_1D)
type(c_ptr), value, intent(in) :: workspace

    integer(c_int64_t) :: i, b, nb, start_i, end_i
    integer(c_int64_t), parameter :: BLOCK_SIZE = 128_c_int64_t
    real(c_double) :: block_A, block_B
    real(c_double), allocatable :: block_A_arr(:), block_B_arr(:)
    real(c_double), allocatable :: pre_A(:), pre_B(:)

    ! Handle trivial case
    if (LEN_1D <= 1_c_int64_t) return

    ! Number of blocks covering indices 2..LEN_1D (the recurrence steps)
    nb = (LEN_1D - 1_c_int64_t + BLOCK_SIZE - 1_c_int64_t) / BLOCK_SIZE

    allocate(block_A_arr(nb), block_B_arr(nb))
    allocate(pre_A(nb+1), pre_B(nb+1))

    ! Compute per‑block transformation (A,B) in parallel
    !$omp parallel default(shared) private(b, start_i, end_i, i, block_A, block_B)
    !$omp do
    do b = 0, nb-1
        start_i = 2_c_int64_t + b * BLOCK_SIZE
        end_i = min(LEN_1D, start_i + BLOCK_SIZE - 1_c_int64_t)
        block_A = 1.0d0
        block_B = 0.0d0
        do i = start_i, end_i
            block_A = block_A * c(i)
            block_B = c(i) * block_B + x(i)
        end do
        block_A_arr(b+1) = block_A
        block_B_arr(b+1) = block_B
    end do
    !$omp end do
    !$omp end parallel

    ! Prefix scan of block transformations (serial, nb is small)
    pre_A(1) = 1.0d0
    pre_B(1) = 0.0d0
    do b = 1, nb
        pre_A(b+1) = block_A_arr(b) * pre_A(b)
        pre_B(b+1) = block_A_arr(b) * pre_B(b) + block_B_arr(b)
    end do

    ! Apply transformations to compute final y values, parallel over blocks
    !$omp parallel default(shared) private(b, start_i, end_i, i, block_A)
    !$omp do
    do b = 0, nb-1
        start_i = 2_c_int64_t + b * BLOCK_SIZE
        end_i = min(LEN_1D, start_i + BLOCK_SIZE - 1_c_int64_t)
        block_A = pre_A(b+1) * y(1) + pre_B(b+1)
        do i = start_i, end_i
            block_A = c(i) * block_A + x(i)
            y(i) = block_A
        end do
    end do
    !$omp end do
    !$omp end parallel

    deallocate(block_A_arr, block_B_arr, pre_A, pre_B)

end subroutine scan_affine_decay_fp64
