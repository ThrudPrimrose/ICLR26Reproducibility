subroutine tsvc_2_s318_fp64(a, result, len_1d, inc) bind(C, name="tsvc_2_s318_fp64")
  ! TSVC s318: result = max(abs(a(i*inc)), i=0..len_1d-1) + (first i attaining it)
  !
  ! The (max, first-index) pair is combined under a total order (larger value
  ! wins, ties to the smaller index), so it is associative: each thread scans
  ! its own contiguous span with 8 independent chains (the update of the
  ! max/index pair is a serial chain a compiler cannot vectorize; 8 chains
  ! hide its latency) and the per-thread pairs are merged after the barrier.
  ! One pass over the data; the per-thread fold uses strict '>', so NaN
  ! values can never enter a partial and the merge stays well defined.
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, inc
  real(c_double), intent(in) :: a((len_1d - 1) * inc + 1)
  real(c_double), intent(out) :: result(1)

  integer(c_int64_t) :: i, n, lo, hi, idx, idx_t, r, cnt
  double precision :: v, maxv, max_t
  double precision :: m1, m2, m3, m4, m5, m6, m7, m8
  double precision :: maxpart(64)
  integer(c_int64_t) :: idxpart(64), j1, j2, j3, j4, j5, j6, j7, j8
  integer :: nt, t, t2

  n = len_1d - 1_c_int64_t
  if (n < 0) then
    result(1) = 0.0d0
    return
  end if
  maxv = abs(a(1))
  idx = 0_c_int64_t

  nt = omp_get_max_threads()
  if (nt > 64) nt = 64

  !$omp parallel private(t, lo, hi, i, v, m1, m2, m3, m4, m5, m6, m7, m8, &
  !$omp                 j1, j2, j3, j4, j5, j6, j7, j8, r, cnt, max_t, idx_t)
    t = omp_get_thread_num()
    lo = (n * t) / nt + 1
    hi = (n * (t + 1)) / nt
    m1 = -huge(1.0d0); j1 = 0_c_int64_t
    m2 = -huge(1.0d0); j2 = 0_c_int64_t
    m3 = -huge(1.0d0); j3 = 0_c_int64_t
    m4 = -huge(1.0d0); j4 = 0_c_int64_t
    m5 = -huge(1.0d0); j5 = 0_c_int64_t
    m6 = -huge(1.0d0); j6 = 0_c_int64_t
    m7 = -huge(1.0d0); j7 = 0_c_int64_t
    m8 = -huge(1.0d0); j8 = 0_c_int64_t
    cnt = hi - lo + 1
    r = mod(cnt, 8_c_int64_t)
    do i = lo, lo + r - 1
      v = abs(a(i * inc + 1))
      if (v > m1) then; m1 = v; j1 = i; end if
    end do
    do i = lo + r, hi, 8_c_int64_t
      v = abs(a(i * inc + 1)); if (v > m1) then; m1 = v; j1 = i; end if
      v = abs(a((i + 1_c_int64_t) * inc + 1)); if (v > m2) then; m2 = v; j2 = i + 1_c_int64_t; end if
      v = abs(a((i + 2_c_int64_t) * inc + 1)); if (v > m3) then; m3 = v; j3 = i + 2_c_int64_t; end if
      v = abs(a((i + 3_c_int64_t) * inc + 1)); if (v > m4) then; m4 = v; j4 = i + 3_c_int64_t; end if
      v = abs(a((i + 4_c_int64_t) * inc + 1)); if (v > m5) then; m5 = v; j5 = i + 4_c_int64_t; end if
      v = abs(a((i + 5_c_int64_t) * inc + 1)); if (v > m6) then; m6 = v; j6 = i + 5_c_int64_t; end if
      v = abs(a((i + 6_c_int64_t) * inc + 1)); if (v > m7) then; m7 = v; j7 = i + 6_c_int64_t; end if
      v = abs(a((i + 7_c_int64_t) * inc + 1)); if (v > m8) then; m8 = v; j8 = i + 7_c_int64_t; end if
    end do
    max_t = m1; idx_t = j1
    if (m2 > max_t) then; max_t = m2; idx_t = j2
    else if (m2 == max_t .and. j2 < idx_t) then; idx_t = j2
    end if
    if (m3 > max_t) then; max_t = m3; idx_t = j3
    else if (m3 == max_t .and. j3 < idx_t) then; idx_t = j3
    end if
    if (m4 > max_t) then; max_t = m4; idx_t = j4
    else if (m4 == max_t .and. j4 < idx_t) then; idx_t = j4
    end if
    if (m5 > max_t) then; max_t = m5; idx_t = j5
    else if (m5 == max_t .and. j5 < idx_t) then; idx_t = j5
    end if
    if (m6 > max_t) then; max_t = m6; idx_t = j6
    else if (m6 == max_t .and. j6 < idx_t) then; idx_t = j6
    end if
    if (m7 > max_t) then; max_t = m7; idx_t = j7
    else if (m7 == max_t .and. j7 < idx_t) then; idx_t = j7
    end if
    if (m8 > max_t) then; max_t = m8; idx_t = j8
    else if (m8 == max_t .and. j8 < idx_t) then; idx_t = j8
    end if
    maxpart(t + 1) = max_t
    idxpart(t + 1) = idx_t
  !$omp barrier
  !$omp end parallel

  do t2 = 0, nt - 1
    if (maxpart(t2 + 1) > maxv) then
      maxv = maxpart(t2 + 1)
      idx = idxpart(t2 + 1)
    else if (maxpart(t2 + 1) == maxv .and. idxpart(t2 + 1) < idx) then
      idx = idxpart(t2 + 1)
    end if
  end do

  result(1) = maxv + dble(idx)
end subroutine tsvc_2_s318_fp64
