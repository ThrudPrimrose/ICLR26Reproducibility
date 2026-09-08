! segment_reduce_ragged: segmented dot product over ragged CSR-style structure.
! out(s) = sum over e in [row_ptr(s), row_ptr(s+1)) of val(e) * w(e).
!
! Segment lengths are ragged (lognormal), so the parallel plan balances the
! real amount of work, not the number of segments: every thread owns a contiguous
! slice of the ELEMENT array [lo, hi) and, via binary search on the monotone
! boundary vector, the set of segments that starts inside that slice.  A segment
! is always reduced by exactly one thread (the one owning its first element),
! so there are no shared writes and the balance error per thread is at most one
! segment length on each side.
module srr_mod
  use iso_c_binding
  implicit none
contains

  subroutine seg_loop(row_ptr, val, w, out, nseg, s_lo, s_hi)
    integer(c_int64_t), intent(in) :: nseg, s_lo, s_hi
    integer(c_int64_t), intent(in) :: row_ptr(nseg + 1)
    real(c_double), intent(in) :: val(nseg * 24)
    real(c_double), intent(in) :: w(nseg * 24)
    real(c_double), intent(out) :: out(nseg)
    integer(c_int64_t) :: s, e
    real(c_double) :: acc

    do s = s_lo, s_hi - 1
      acc = 0.0d0
      !$omp simd reduction(+:acc)
      do e = row_ptr(s) + 1, row_ptr(s + 1)
        acc = acc + val(e) * w(e)
      end do
      out(s) = acc
    end do
  end subroutine seg_loop

  pure function lower_bound(arr, n, x) result(res)
    integer(c_int64_t), intent(in) :: n, x
    integer(c_int64_t), intent(in) :: arr(n)
    integer(c_int64_t) :: res, m, r, mid
    m = 1
    r = n
    do while (m < r)
      mid = (m + r) / 2
      if (arr(mid) < x) then
        m = mid + 1
      else
        r = mid
      end if
    end do
    res = m
  end function lower_bound

end module srr_mod

subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, nseg, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  use srr_mod
  implicit none
  integer(c_int64_t), value, intent(in) :: nseg
  integer(c_int64_t), intent(in) :: row_ptr(nseg + 1)
  real(c_double), intent(in) :: val(nseg * 24)
  real(c_double), intent(in) :: w(nseg * 24)
  real(c_double), intent(out) :: out(nseg)
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: total, lo, hi, s_lo, s_hi
  integer :: nt, t

  if (nseg <= 0) return
  total = nseg * 24_8

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1

  if (nt == 1 .or. total < 16384_8) then
    call seg_loop(row_ptr, val, w, out, nseg, 0_8, nseg + 1)
  else
    !$omp parallel private(t, lo, hi, s_lo, s_hi)
    t = omp_get_thread_num()
    lo = (total * t) / nt
    hi = (total * (t + 1)) / nt
    s_lo = lower_bound(row_ptr, nseg + 1, lo)
    s_hi = lower_bound(row_ptr, nseg + 1, hi)
    call seg_loop(row_ptr, val, w, out, nseg, s_lo, s_hi)
    !$omp end parallel
  end if
end subroutine segment_reduce_ragged_fp64
