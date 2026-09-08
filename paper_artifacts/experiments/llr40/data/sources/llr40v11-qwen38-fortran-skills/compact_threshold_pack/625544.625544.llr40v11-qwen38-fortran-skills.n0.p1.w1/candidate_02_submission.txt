subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, len_1d, &
    workspace, workspace_size) bind(C)
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  integer(c_int64_t), intent(inout) :: out_count(1)
  real(c_double), intent(inout) :: packed(len_1d)
  real(c_double), intent(in) :: src(len_1d)
  real(c_double), intent(in) :: weight(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i, n, lo, hi, run, t, nt, k
  integer(c_int64_t) :: part(512)
  integer(c_int64_t) :: base(513)

  n = len_1d

  if (n < 65536) then
    run = 0
    do i = 1, n
      if (src(i) > 0.0d0) then
        packed(run + 1) = src(i) * weight(i)
        run = run + 1
      end if
    end do
    out_count(1) = run
    return
  end if

  !$omp parallel shared(part, base) private(t, lo, hi, run, i, k)
  t = omp_get_thread_num()
  nt = omp_get_num_threads()
  lo = (n * t) / nt + 1
  hi = (n * (t + 1)) / nt
  run = 0
  !$omp simd reduction(+:run)
  do i = lo, hi
    if (src(i) > 0.0d0) run = run + 1
  end do
  part(t + 1) = run
  !$omp barrier
  !$omp single
  base(1) = 0
  do k = 1, nt
    base(k + 1) = base(k) + part(k)
  end do
  !$omp end single
  !$omp barrier
  run = 0
  do i = lo, hi
    if (src(i) > 0.0d0) then
      packed(base(t + 1) + run + 1) = src(i) * weight(i)
      run = run + 1
    end if
  end do
  !$omp end parallel
  out_count(1) = base(nt + 1)
end subroutine compact_threshold_pack_fp64
