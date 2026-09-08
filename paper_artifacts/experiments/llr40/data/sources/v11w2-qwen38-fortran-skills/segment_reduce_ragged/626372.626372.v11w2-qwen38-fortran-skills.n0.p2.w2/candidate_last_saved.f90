subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, workspace, workspace_size) bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: NSEG
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: out(NSEG)
  integer(c_int64_t), intent(in) :: row_ptr(NSEG + 1)
  real(c_double), intent(in) :: val(NSEG * 24)
  real(c_double), intent(in) :: w(NSEG * 24)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: TOT, nt, t, elo, ehi, s, e, lo, hi, mid
  real(c_double) :: acc

  if (NSEG <= 0) return
  TOT = row_ptr(NSEG + 1)
  nt = omp_get_max_threads()
  if (nt < 1) nt = 1

  !$omp parallel default(none) shared(NSEG, TOT, nt, out, row_ptr, val, w) private(t, elo, ehi, s, e, lo, hi, mid, acc)
  t = omp_get_thread_num()
  elo = TOT * t / nt
  ehi = TOT * (t + 1) / nt
  if (ehi > elo) then
    ! first segment s with row_ptr(s) >= elo (sentinel NSEG+1 when none)
    lo = 1
    hi = NSEG + 1
    do while (lo < hi)
      mid = (lo + hi) / 2
      if (row_ptr(mid) >= elo) then
        hi = mid
      else
        lo = mid + 1
      end if
    end do
    s = lo
    do while (s <= NSEG .and. row_ptr(s) < ehi)
      acc = 0.0d0
      !$omp simd reduction(+:acc)
      do e = row_ptr(s) + 1, row_ptr(s + 1)
        acc = acc + val(e) * w(e)
      end do
      out(s) = acc
      s = s + 1
    end do
    if (t == nt - 1) then
      do while (s <= NSEG .and. row_ptr(s) == ehi)
        out(s) = 0.0d0
        s = s + 1
      end do
    end if
  end if
  !$omp end parallel
end subroutine segment_reduce_ragged_fp64
