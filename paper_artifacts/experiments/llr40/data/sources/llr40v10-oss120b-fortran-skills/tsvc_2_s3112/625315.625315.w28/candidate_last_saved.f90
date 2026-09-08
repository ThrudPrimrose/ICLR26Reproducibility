subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  integer(c_int64_t) :: i, start_idx, end_idx, chunk
  integer :: tid, nthreads
  real(c_double), allocatable :: block_sum(:), block_offset(:)
  real(c_double) :: sum_local

  nthreads = omp_get_max_threads()
  allocate(block_sum(nthreads), block_offset(nthreads))
  block_sum = 0.0_c_double
  block_offset = 0.0_c_double

  ! First pass: compute sum of each block
  !$omp parallel private(tid, start_idx, end_idx, sum_local, chunk) shared(block_sum)
    tid = omp_get_thread_num()
    chunk = (LEN_1D + nthreads - 1) / nthreads
    start_idx = tid * chunk + 1
    end_idx = min(start_idx + chunk - 1, LEN_1D)
    sum_local = 0.0_c_double
    if (start_idx <= end_idx) then
      do i = start_idx, end_idx
        sum_local = sum_local + a(i)
      end do
    end if
    block_sum(tid+1) = sum_local
  !$omp end parallel

  ! Compute offsets (prefix sum of block sums)
  block_offset(1) = 0.0_c_double
  do i = 2, nthreads
    block_offset(i) = block_offset(i-1) + block_sum(i-1)
  end do

  ! Second pass: compute prefix sum within each block using offset
  !$omp parallel private(tid, start_idx, end_idx, i, sum_local, chunk) shared(block_offset)
    tid = omp_get_thread_num()
    chunk = (LEN_1D + nthreads - 1) / nthreads
    start_idx = tid * chunk + 1
    end_idx = min(start_idx + chunk - 1, LEN_1D)
    sum_local = block_offset(tid+1)
    if (start_idx <= end_idx) then
      do i = start_idx, end_idx
        sum_local = sum_local + a(i)
        b(i) = sum_local
      end do
    end if
  !$omp end parallel

  deallocate(block_sum, block_offset)
end subroutine tsvc_2_s3112_fp64
