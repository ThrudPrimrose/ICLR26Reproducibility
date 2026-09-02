subroutine tsvc_2_s318_fp64(a, result, len_1d, inc, workspace, workspace_size) bind(C, name="tsvc_2_s318_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, inc, workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(inout) :: result(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: n, lo, hi, i, idx, besti
  integer :: nt, t
  real(c_double) :: m, v1, v2, v3, v4, v5, v6, v7, v8, pa, pm, bm, qa, qm, bm2, bx, bestv
  real(c_double), allocatable :: tv(:)
  integer(c_int64_t), allocatable :: ti(:)
  integer(c_int64_t) :: ja, pj, ka, kj, bj, bj2, jx

  n = len_1d
  if (n == 1_c_int64_t) then
    result(1) = abs(a(1))
    return
  end if
  if (n <= 0_c_int64_t) return
  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  allocate (tv(nt), ti(nt))
  !$omp parallel shared(a, n, tv, ti, nt) private(t, lo, hi, i, idx, m, v1, v2, v3, v4, v5, v6, v7, v8, pa, pm, bm, qa, qm, bm2, bx, ja, pj, ka, kj, bj, bj2, jx)
    t = omp_get_thread_num()
    lo = (n * t) / nt + 1_c_int64_t
    hi = (n * (t + 1_c_int64_t)) / nt
    m = -1.0d0
    idx = 0_c_int64_t
    i = lo
    do while (i < hi .and. mod(i - 1_c_int64_t, 4_c_int64_t) /= 0_c_int64_t)
      v1 = abs(a(i))
      if (v1 > m) then
        m = v1
        idx = i
      end if
      i = i + 1_c_int64_t
    end do
    do while (i + 3_c_int64_t <= hi)
      v1 = abs(a(i))
      v2 = abs(a(i + 1_c_int64_t))
      v3 = abs(a(i + 2_c_int64_t))
      v4 = abs(a(i + 3_c_int64_t))
      if (v1 >= v2) then
        pa = v1
        ja = i
      else
        pa = v2
        ja = i + 1_c_int64_t
      end if
      if (v3 >= v4) then
        pm = v3
        pj = i + 2_c_int64_t
      else
        pm = v4
        pj = i + 3_c_int64_t
      end if
      if (pa >= pm) then
        bm = pa
        bj = ja
      else
        bm = pm
        bj = pj
      end if
      if (bm > m) then
        m = bm
        idx = bj
      end if
      i = i + 4_c_int64_t
    end do
    do while (i <= hi)
      v1 = abs(a(i))
      if (v1 > m) then
        m = v1
        idx = i
      end if
      i = i + 1_c_int64_t
    end do
    tv(t + 1) = m
    ti(t + 1) = idx
  !$omp end parallel
  bestv = -1.0d0
  besti = 0_c_int64_t
  do t = 1, nt
    if (tv(t) > bestv .or. (tv(t) == bestv .and. (besti == 0_c_int64_t .or. ti(t) < besti))) then
      bestv = tv(t)
      besti = ti(t)
    end if
  end do
  result(1) = bestv + dble(besti - 1)
end subroutine
