subroutine scan_affine_decay_fp64(y, c, x, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: y(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: x(LEN_1D)
  integer(c_int64_t) :: i, block, nb, block_size
  integer(c_int64_t) :: s, e
  real(c_double) :: a, b, tmp
  real(c_double), allocatable :: block_A(:), block_B(:), start_y(:)
  ! Ensure the first element (seed) is correctly initialized.
  if (LEN_1D >= 1) then
    y(1) = x(1)
  end if
  if (LEN_1D <= 1) return

  ! Choose a block size; tune for cache and parallelism.
  block_size = 1024_c_int64_t
  nb = (LEN_1D - 1 + block_size - 1) / block_size

  allocate(block_A(nb), block_B(nb), start_y(nb))

  ! Phase 1: compute block affine transforms (A,B) in parallel.
  !$omp parallel do private(block, s, e, i, a, b) schedule(static)
  do block = 1, nb
    s = 2 + (block-1) * block_size
    e = min(LEN_1D, s + block_size - 1)
    a = 1.0_c_double
    b = 0.0_c_double
    do i = s, e
      a = a * c(i)
      b = b * c(i) + x(i)
    end do
    block_A(block) = a
    block_B(block) = b
  end do
  !$omp end parallel do

  ! Phase 2: sequential prefix to obtain the starting y for each block.
  start_y(1) = y(1)
  do block = 2, nb
    start_y(block) = block_A(block-1) * start_y(block-1) + block_B(block-1)
  end do

  ! Phase 3: compute final y values in each block using the starting value.
  !$omp parallel do private(block, s, e, i, tmp) schedule(static)
  do block = 1, nb
    s = 2 + (block-1) * block_size
    e = min(LEN_1D, s + block_size - 1)
    tmp = start_y(block)
    do i = s, e
      tmp = c(i) * tmp + x(i)
      y(i) = tmp
    end do
  end do
  !$omp end parallel do

  deallocate(block_A, block_B, start_y)
end subroutine scan_affine_decay_fp64
