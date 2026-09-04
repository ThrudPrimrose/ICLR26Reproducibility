subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, LEN_1D, workspace, workspace_bytes) bind(C, name="compact_threshold_pack_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   ! Arguments
   integer(c_int64_t), intent(out) :: out_count(1)
   real(c_double), intent(in) :: src(*)
   real(c_double), intent(in) :: weight(*)
   real(c_double), intent(out) :: packed(*)
   integer(c_int64_t), value :: LEN_1D
   type(c_ptr), value :: workspace   ! unused
   integer(c_int64_t), value :: workspace_bytes   ! unused

   integer(c_int) :: max_threads, nthreads_use
   integer(c_int64_t) :: nthreads, tid
   integer(c_int64_t) :: i, i_start, i_end, local_count, pos, total_n, base, rem
   integer(c_int64_t), allocatable :: block_counts(:)
   integer(c_int64_t), allocatable :: block_offsets(:)

   if (LEN_1D <= 0_c_int64_t) then
      out_count(1) = 0_c_int64_t
      return
   end if
   ! Allocate per‑thread buffers based on the maximum number of threads
   max_threads = omp_get_max_threads()
   ! Determine usable number of threads (no more than LEN_1D)
   if (LEN_1D < int(max_threads, kind=c_int64_t)) then
      nthreads_use = int(LEN_1D)
   else
      nthreads_use = max_threads
   end if
   ! Ensure at least one thread for positive LEN_1D
   if (nthreads_use < 1_c_int) nthreads_use = 1_c_int
   call omp_set_num_threads(nthreads_use)
   ! Allocate per‑thread count and offset arrays
   ! allocate(block_counts(nthreads_use))
   ! allocate(block_offsets(nthreads_use))
   ! block_counts = 0_c_int64_t
   
   !$omp parallel private(tid, i, i_start, i_end, local_count, pos)
   tid = omp_get_thread_num()
   nthreads = omp_get_num_threads()
	!$omp single
	   allocate(block_counts(nthreads))
	   allocate(block_offsets(nthreads))
	   block_counts = 0_c_int64_t
	!$omp end single

   ! Compute this thread's slice (1‑based indexing)
   		base = LEN_1D / nthreads
	rem = LEN_1D - base * nthreads
	i_start = tid * base + min(tid, rem) + 1_c_int64_t
	i_end = i_start + base - 1_c_int64_t
	if (tid < rem) i_end = i_end + 1_c_int64_t

   ! First pass: count survivors in this slice
   local_count = 0_c_int64_t
   do i = i_start, i_end
      if (src(i) > 0.0_c_double) then
         local_count = local_count + 1_c_int64_t
      end if
   end do
   block_counts(tid+1) = local_count
   !$omp barrier

   ! Compute prefix offsets (single thread)
   !$omp single
      block_offsets(1) = 0_c_int64_t
      do i = 2, nthreads
         block_offsets(i) = block_offsets(i-1) + block_counts(i-1)
      end do
      total_n = block_offsets(nthreads) + block_counts(nthreads)
   !$omp end single

   ! Second pass: write packed values using the computed offsets
   pos = block_offsets(tid+1)
   do i = i_start, i_end
      if (src(i) > 0.0_c_double) then
         packed(pos+1) = src(i) * weight(i)
         pos = pos + 1_c_int64_t
      end if
   end do
   !$omp barrier

   ! Store final count and zero tail (single thread)
   !$omp single
      out_count(1) = total_n
      do i = total_n + 1, LEN_1D
         packed(i) = 0.0_c_double
      end do
   !$omp end single
   !$omp end parallel

   deallocate(block_counts)
   deallocate(block_offsets)
end subroutine compact_threshold_pack_fp64
