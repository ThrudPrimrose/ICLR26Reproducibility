module compact_threshold_pack_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_int8_t
  use omp_lib
  implicit none

  integer(c_int64_t), allocatable, save :: counts(:), offsets(:)
contains

  subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_bytes) bind(c, name="compact_threshold_pack_fp64")
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t), intent(out) :: out_count
    real(c_double), intent(out) :: packed(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    real(c_double), intent(in) :: weight(LEN_1D)
    integer(c_int8_t), intent(inout) :: workspace(*)
    integer(c_int64_t), value :: workspace_bytes

    integer(c_int64_t) :: i, n, cnt, start, fin, off
    integer :: tid, nt, j
    integer(c_int64_t) :: chunk, nthreads

    n = LEN_1D

    chunk = 0_c_int64_t
    nthreads = 0_c_int64_t

    if (n <= 0_c_int64_t) then
      out_count = 0_c_int64_t
      return
    end if

    ! Small inputs: sequential compaction to avoid OpenMP overhead.
    if (n < 4096_c_int64_t) then
      cnt = 0_c_int64_t
      do i = 1_c_int64_t, n
        if (src(i) > 0.0_c_double) then
          cnt = cnt + 1_c_int64_t
          packed(cnt) = src(i) * weight(i)
        end if
      end do
      out_count = cnt
      return
    end if

    ! Persistent scratch for per-thread counts and prefix offsets.
    nt = omp_get_max_threads()
    if (.not. allocated(counts)) then
      allocate(counts(0:nt), offsets(0:nt))
    else if (ubound(counts, 1) < nt) then
      deallocate(counts, offsets)
      allocate(counts(0:nt), offsets(0:nt))
    end if

    !$omp parallel default(none) shared(src, weight, packed, counts, offsets, n, out_count, nthreads, chunk) private(tid, i, start, fin, cnt, off, j)
    tid = omp_get_thread_num()
    if (tid == 0) then
      nthreads = int(omp_get_num_threads(), c_int64_t)
      chunk = (n + nthreads - 1_c_int64_t) / nthreads
    end if
    !$omp barrier

    start = int(tid, c_int64_t) * chunk + 1_c_int64_t
    fin = min(n, start + chunk - 1_c_int64_t)

    cnt = 0_c_int64_t
    do i = start, fin
      if (src(i) > 0.0_c_double) cnt = cnt + 1_c_int64_t
    end do
    counts(tid) = cnt

    !$omp barrier
    if (tid == 0) then
      offsets(0) = 0_c_int64_t
      do j = 1, int(nthreads) - 1
        offsets(j) = offsets(j - 1) + counts(j - 1)
      end do
      offsets(int(nthreads)) = offsets(int(nthreads) - 1) + counts(int(nthreads) - 1)
      out_count = offsets(int(nthreads))
    end if
    !$omp barrier

    off = offsets(tid)
    cnt = 0_c_int64_t
    do i = start, fin
      if (src(i) > 0.0_c_double) then
        packed(off + cnt + 1_c_int64_t) = src(i) * weight(i)
        cnt = cnt + 1_c_int64_t
      end if
    end do
    !$omp end parallel
  end subroutine compact_threshold_pack_fp64

end module compact_threshold_pack_mod
