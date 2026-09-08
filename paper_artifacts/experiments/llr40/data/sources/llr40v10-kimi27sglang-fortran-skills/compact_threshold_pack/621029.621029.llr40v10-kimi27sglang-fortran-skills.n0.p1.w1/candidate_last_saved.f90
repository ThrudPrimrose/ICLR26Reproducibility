subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  integer(c_int64_t), intent(out) :: out_count
  real(c_double), intent(inout) :: packed(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  real(c_double), intent(in) :: weight(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i
  integer(c_int64_t) :: nt, t, lo, hi, cnt
  integer(c_int64_t), allocatable :: offsets(:)

  nt = omp_get_max_threads()
  allocate(offsets(0:nt))

  !$omp parallel private(t, lo, hi, cnt, i)
  t = omp_get_thread_num()
  lo = (LEN_1D * t) / nt + 1
  hi = (LEN_1D * (t + 1)) / nt

  cnt = 0
  !$omp simd reduction(+:cnt)
  do i = lo, hi
    if (src(i) > 0.0d0) cnt = cnt + 1
  end do
  !$omp end simd
  offsets(t + 1) = cnt

  !$omp barrier

  !$omp single
  offsets(0) = 0
  do i = 1, nt
    offsets(i) = offsets(i) + offsets(i - 1)
  end do
  !$omp end single

  cnt = offsets(t)
  do i = lo, hi
    if (src(i) > 0.0d0) then
      cnt = cnt + 1
      packed(cnt) = src(i) * weight(i)
    end if
  end do
  !$omp end parallel

  out_count = offsets(nt)
  deallocate(offsets)
end subroutine compact_threshold_pack_fp64
