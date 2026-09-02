subroutine tsvc_2_s1232_fp64(aa, bb, cc, len_2d, vlen, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d, vlen, workspace_size
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d), cc(len_2d, len_2d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: n, v, j, r, t, nt, lo, hi, tot, c0, c1, a0, a1, mid

  n = len_2d
  v = vlen
  if (n <= 0) return

  ! flat offset of row i (rows 1..n, columns contiguous): exclusive prefix sum of row lengths
  ! L(r) = n if v == 0, else (r-1)/v + 1
  a0 = 0
  a1 = 1
  ! total work
  if (v == 0) then
    tot = n * n
  else
    mid = n - 1
    a0 = mid / v
    a1 = mid - a0 * v
    tot = n + v * a0 * (a0 - 1) / 2 + a0 * (a1 + 1)
  end if

  !$omp parallel shared(aa, bb, cc, n, v, tot) private(t, nt, lo, hi, r, c0, c1, j, a0, a1, mid)
  nt = omp_get_max_threads()
  t = omp_get_thread_num()
  lo = tot * t / nt
  hi = tot * (t + 1) / nt
  if (lo < hi) then
    ! binary search: a0 = largest row index with S(a0) <= lo
    a0 = 1
    a1 = n + 1
    do while (a1 - a0 > 1)
      mid = (a0 + a1) / 2
      if (sval(mid) <= lo) then
        a0 = mid
      else
        a1 = mid
      end if
    end do
    r = a0
    c0 = lo - sval(a0) + 1
    do while (r <= n .and. sval(r) < hi)
      c1 = lval(r)
      c1 = min(c1, hi - sval(r))
      if (c1 >= c0) then
        !$omp simd
        do j = c0, c1
          aa(j, r) = bb(j, r) + cc(j, r)
        end do
      end if
      r = r + 1
      c0 = 1
    end do
  end if
  !$omp end parallel

contains

  integer(c_int64_t) function sval(i)
    integer(c_int64_t), intent(in) :: i
    integer(c_int64_t) :: m, q, rr
    if (i <= 1) then
      sval = 0
      return
    end if
    if (v == 0) then
      sval = (i - 1) * n
      return
    end if
    m = i - 2
    q = m / v
    rr = m - q * v
    sval = (i - 1) + v * q * (q - 1) / 2 + q * (rr + 1)
  end function sval

  integer(c_int64_t) function lval(r)
    integer(c_int64_t), intent(in) :: r
    if (v == 0) then
      lval = n
    else
      lval = (r - 1) / v + 1
    end if
  end function lval

end subroutine tsvc_2_s1232_fp64
