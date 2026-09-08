module scan_affine_decay_mod
  use iso_c_binding
  implicit none
contains
  subroutine scan_affine_decay_fp64(c, x, y, LEN_1D) bind(C, name="scan_affine_decay_fp64")
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: c(*), x(*)
    real(c_double), intent(inout) :: y(*)
    integer(c_int64_t) :: i, k, s, e, M
    integer(c_int64_t) :: N, B
    real(c_double) :: local_A, local_B, t
    real(c_double), allocatable :: block_A(:), block_B(:), block_start_y(:)

    if (LEN_1D <= 0_c_int64_t) return
    ! Set seed (should already be set by initializer)
    y(1) = x(1)

    N = LEN_1D - 1_c_int64_t
    B = 1024_c_int64_t
    M = (N + B - 1_c_int64_t) / B
    allocate(block_A(M), block_B(M), block_start_y(M))

    ! Compute block transforms (A, B) for each block in parallel
    !$omp parallel do private(k, s, e, i, local_A, local_B) shared(block_A, block_B, c, x)
    do k = 1, M
      s = 2 + (k-1) * B
      e = min(LEN_1D, s + B - 1)
      local_A = 1.0_c_double
      local_B = 0.0_c_double
      do i = s, e
        local_B = c(i) * local_B + x(i)
        local_A = local_A * c(i)
      end do
      block_A(k) = local_A
      block_B(k) = local_B
    end do
    !$omp end parallel do

    ! Prefix over blocks to obtain starting y for each block
    t = y(1)
    do k = 1, M
      block_start_y(k) = t
      t = block_A(k) * t + block_B(k)
    end do

    ! Compute final y values within each block in parallel
    !$omp parallel do private(k, s, e, i, t) shared(block_start_y, c, x, y)
    do k = 1, M
      s = 2 + (k-1) * B
      e = min(LEN_1D, s + B - 1)
      t = block_start_y(k)
      do i = s, e
        t = c(i) * t + x(i)
        y(i) = t
      end do
    end do
    !$omp end parallel do

    deallocate(block_A, block_B, block_start_y)
  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
