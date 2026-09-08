module tsvc_2_s3112_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: b(*)
    integer(c_int64_t), value :: LEN_1D
    integer :: max_threads, tid, nthreads
    integer(c_int64_t) :: i, start, end_, chunk
    real(c_double), allocatable :: block_sum(:), offset_arr(:)
    real(c_double) :: sum_local, offset

    max_threads = omp_get_max_threads()
    allocate(block_sum(max_threads))
    allocate(offset_arr(max_threads))

    ! First parallel region: compute the sum of each block
    !$omp parallel private(tid, nthreads, chunk, start, end_, i, sum_local)
      tid = omp_get_thread_num()
      nthreads = omp_get_num_threads()
      chunk = (LEN_1D + nthreads - 1) / nthreads
      start = tid * chunk + 1
      end_ = min(start + chunk - 1, LEN_1D)
      sum_local = 0.0_c_double
      if (start <= LEN_1D) then
        do i = start, end_
          sum_local = sum_local + a(i)
        end do
      end if
      block_sum(tid+1) = sum_local
    !$omp end parallel

    ! Compute offsets for each block (sequential prefix of block sums)
    offset = 0.0_c_double
    do i = 1, max_threads
      offset_arr(i) = offset
      offset = offset + block_sum(i)
    end do

    ! Second parallel region: compute prefix sums for each block using the offset
    !$omp parallel private(tid, nthreads, chunk, start, end_, i, sum_local)
      tid = omp_get_thread_num()
      nthreads = omp_get_num_threads()
      chunk = (LEN_1D + nthreads - 1) / nthreads
      start = tid * chunk + 1
      end_ = min(start + chunk - 1, LEN_1D)
      sum_local = offset_arr(tid+1)
      if (start <= LEN_1D) then
        do i = start, end_
          sum_local = sum_local + a(i)
          b(i) = sum_local
        end do
      end if
    !$omp end parallel

    deallocate(block_sum)
    deallocate(offset_arr)
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112_mod
