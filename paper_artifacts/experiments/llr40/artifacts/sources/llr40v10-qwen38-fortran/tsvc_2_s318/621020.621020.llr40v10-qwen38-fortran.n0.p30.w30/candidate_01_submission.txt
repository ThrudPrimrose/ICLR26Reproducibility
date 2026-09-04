subroutine tsvc_2_s318_fp64(a, result, n, inc) bind(c, name="tsvc_2_s318_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in), dimension(*) :: a
  real(c_double), intent(out), dimension(*) :: result
  integer(c_int64_t), value :: n
  integer(c_int64_t), value :: inc

  integer(c_int64_t) :: i, k, j, lo, hi, ch, t
  integer(c_int64_t) :: nt, mt, B
  real(c_double) :: m, bm, v
  integer(c_int64_t) :: mi
  logical :: badg
  real(c_double), allocatable :: tm(:)
  integer(c_int64_t), allocatable :: ti(:)

  if (n <= 0) then
    result(1) = 0.0d0
    return
  end if
  if (n == 1 .or. inc <= 0) then
    result(1) = abs(a(1)) + 0.0d0
    return
  end if

  mt = omp_get_max_threads()
  if (mt < 1) mt = 1
  allocate(tm(mt), ti(mt))

  if (mt > 1 .and. n >= 65536) then
    badg = .false.
    !$omp parallel shared(a, n, inc, nt, tm, ti, badg) private(t, lo, hi, ch, m, bm, mi, k, j, i, v, B)
      t = omp_get_thread_num()
      nt = omp_get_num_threads()
      ch = (n + nt - 1) / nt
      lo = t * ch
      hi = min((t + 1) * ch, n)
      if (lo >= hi) then
        tm(t + 1) = -1.0d300
        ti(t + 1) = -1
      else if (inc == 1) then
        m = -1.0d300
        mi = -1
        B = 4096
        do k = lo, hi - 1, B
          j = min(k + B, hi)
          bm = maxval(abs(a(k + 1:j)))
          if (bm > m) then
            if (bm == bm) then
              m = bm
              do i = k, j - 1
                if (abs(a(i + 1)) == bm) exit
              end do
              mi = i
            else
              !$omp critical
              badg = .true.
              !$omp end critical
            end if
          end if
        end do
        tm(t + 1) = m
        ti(t + 1) = mi
      else
        m = -1.0d300
        mi = -1
        B = 4096
        do k = lo, hi - 1, B
          j = min(k + B, hi)
          bm = -1.0d300
          do i = k, j - 1
            bm = max(bm, abs(a(i * inc + 1)))
          end do
          if (bm > m) then
            if (bm == bm) then
              m = bm
              do i = k, j - 1
                if (abs(a(i * inc + 1)) == bm) exit
              end do
              mi = i
            else
              !$omp critical
              badg = .true.
              !$omp end critical
            end if
          end if
        end do
        tm(t + 1) = m
        ti(t + 1) = mi
      end if
    !$omp end parallel
    if (.not. badg) then
      m = -1.0d300
      mi = 0
      do t = 0, nt - 1
        if (tm(t + 1) > m) then
          m = tm(t + 1)
          mi = ti(t + 1)
        end if
      end do
      result(1) = m + dble(mi)
      deallocate(tm, ti)
      return
    end if
  end if
  deallocate(tm, ti)

  ! serial reference (fallback + small n)
  m = abs(a(1))
  mi = 0
  do i = 1, n - 1
    v = abs(a(i * inc + 1))
    if (v > m) then
      m = v
      mi = i
    end if
  end do
  result(1) = m + dble(mi)
end subroutine tsvc_2_s318_fp64
