module compact_threshold_pack_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  ! Stream compaction: for each i where src[i] > 0, write src[i] * weight[i] to packed in order,
  ! and store the total number of survivors in out_count[0].
  ! Expected C signature (due to canonical binding order):
  !   void compact_threshold_pack_fp64(int64_t *out_count, double *packed,
  !                                   const double *src, const double *weight,
  !                                   const int64_t LEN_1D,
  !                                   uint8_t *workspace, const int64_t workspace_bytes);
  ! The workspace arguments are ignored here but are required by the ABI.
  subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_bytes) bind(C, name="compact_threshold_pack_fp64")
    integer(c_int64_t), intent(out) :: out_count(1)
    real(c_double), intent(out) :: packed(*)
    real(c_double), intent(in) :: src(*)
    real(c_double), intent(in) :: weight(*)
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer(c_int64_t) :: i, tid, nthreads
    integer(c_int64_t) :: chunk_start, chunk_end
    integer(c_int64_t), allocatable :: thread_counts(:)
    integer(c_int64_t), allocatable :: offsets(:)
    integer(c_int64_t) :: local_sum, local_idx, total

    ! Determine number of threads to use
    nthreads = omp_get_max_threads()
    allocate(thread_counts(0:nthreads-1))
    thread_counts = 0_c_int64_t

    ! First parallel region: count survivors per thread
    !$omp parallel private(tid, i, chunk_start, chunk_end, local_sum) shared(thread_counts)
      tid = omp_get_thread_num()
      chunk_start = tid * (LEN_1D / nthreads) + 1
      chunk_end = (tid + 1) * (LEN_1D / nthreads)
      if (tid == nthreads - 1) chunk_end = LEN_1D
      local_sum = 0_c_int64_t
      do i = chunk_start, chunk_end
        if (src(i) > 0.0_c_double) then
          local_sum = local_sum + 1_c_int64_t
        end if
      end do
      thread_counts(tid) = local_sum
    !$omp end parallel

    ! Compute exclusive prefix sum of thread_counts to get offsets
    allocate(offsets(0:nthreads))
    offsets(0) = 0_c_int64_t
    do i = 1, nthreads
      offsets(i) = offsets(i-1) + thread_counts(i-1)
    end do
    total = offsets(nthreads)

    ! Second parallel region: write packed values using computed offsets
    !$omp parallel private(tid, i, chunk_start, chunk_end, local_idx) shared(offsets)
      tid = omp_get_thread_num()
      chunk_start = tid * (LEN_1D / nthreads) + 1
      chunk_end = (tid + 1) * (LEN_1D / nthreads)
      if (tid == nthreads - 1) chunk_end = LEN_1D
      local_idx = offsets(tid)
      do i = chunk_start, chunk_end
        if (src(i) > 0.0_c_double) then
          local_idx = local_idx + 1_c_int64_t
          packed(local_idx) = src(i) * weight(i)
        end if
      end do
    !$omp end parallel

    out_count(1) = total

    deallocate(thread_counts)
    deallocate(offsets)
  end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_mod
