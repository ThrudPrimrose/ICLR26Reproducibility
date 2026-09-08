module tsvc_2_s3112_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
    ! Arguments from C
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: b(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, start_i, end_i, chunk, nthreads, tid
    real(c_double) :: sum_local
    real(c_double), allocatable :: partial_sums(:)
    real(c_double) :: running
    if (LEN_1D <= 0_c_int64_t) then
      return
    end if
    nthreads = omp_get_max_threads()
    allocate(partial_sums(nthreads))
    partial_sums = 0.0_c_double

    !$omp parallel private(tid, start_i, end_i, chunk, sum_local, i) shared(a, b, LEN_1D, nthreads, partial_sums)
      tid = omp_get_thread_num()
      chunk = (LEN_1D + nthreads - 1) / nthreads
      start_i = tid * chunk + 1_c_int64_t
      end_i = start_i + chunk - 1_c_int64_t
      if (end_i > LEN_1D) end_i = LEN_1D

      ! Compute sum of this chunk
      sum_local = 0.0_c_double
      do i = start_i, end_i
        sum_local = sum_local + a(i)
      end do
      partial_sums(tid+1) = sum_local

      !$omp barrier

      !$omp single
        running = 0.0_c_double
        do i = 1, nthreads
          sum_local = partial_sums(i)
          partial_sums(i) = running
          running = running + sum_local
        end do
      !$omp end single

      sum_local = partial_sums(tid+1)
      do i = start_i, end_i
        sum_local = sum_local + a(i)
        b(i) = sum_local
      end do
    !$omp end parallel

    deallocate(partial_sums)
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112_mod
