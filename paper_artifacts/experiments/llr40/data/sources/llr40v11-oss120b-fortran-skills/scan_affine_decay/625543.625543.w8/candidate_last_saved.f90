subroutine scan_affine_decay_fp64(y, c, x, LEN_1D) bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: y(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: x(LEN_1D)
  integer(c_int64_t) :: i, b
  integer(c_int64_t), parameter :: blockSize = 256_c_int64_t
  integer(c_int64_t) :: blockCount, startIdx, endIdx
  real(c_double), allocatable :: block_A(:), block_B(:)
  real(c_double), allocatable :: prefix_A(:), prefix_B(:)
  real(c_double) :: y_val

  if (LEN_1D <= 1_c_int64_t) then
    return
  end if

  ! Ensure seed value is set
  y(1) = x(1)

  blockCount = ((LEN_1D - 1_c_int64_t) + blockSize - 1_c_int64_t) / blockSize

  allocate(block_A(blockCount))
  allocate(block_B(blockCount))
  allocate(prefix_A(blockCount+1))
  allocate(prefix_B(blockCount+1))

  !$omp parallel do private(b, startIdx, endIdx, i) shared(block_A, block_B, c, x)
  do b = 0, blockCount-1
    startIdx = 2_c_int64_t + b*blockSize
    endIdx = min(startIdx + blockSize - 1_c_int64_t, LEN_1D)
    block_A(b+1) = 1.0_c_double
    block_B(b+1) = 0.0_c_double
    do i = startIdx, endIdx
      block_A(b+1) = block_A(b+1) * c(i)
      block_B(b+1) = c(i) * block_B(b+1) + x(i)
    end do
  end do
  !$omp end parallel do

  prefix_A(1) = 1.0_c_double
  prefix_B(1) = 0.0_c_double
  do b = 1, blockCount
    prefix_A(b+1) = block_A(b) * prefix_A(b)
    prefix_B(b+1) = block_A(b) * prefix_B(b) + block_B(b)
  end do

  !$omp parallel do private(b, i, startIdx, endIdx, y_val) shared(y, c, x, prefix_A, prefix_B)
  do b = 0, blockCount-1
    startIdx = 2_c_int64_t + b*blockSize
    endIdx = min(startIdx + blockSize - 1_c_int64_t, LEN_1D)
        y_val = prefix_A(b+1) * y(1) + prefix_B(b+1)
    do i = startIdx, endIdx
      y_val = c(i) * y_val + x(i)
      y(i) = y_val
    end do
  end do
  !$omp end parallel do

  deallocate(block_A, block_B, prefix_A, prefix_B)

end subroutine scan_affine_decay_fp64
