module amod
  use iso_c_binding
  implicit none
  real(c_double), allocatable, save :: bmax(:)
  integer(c_int64_t), save :: bmax_cap = 0
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none
    integer(c_int64_t), value, intent(in) :: len_1d
    real(c_double), intent(in) :: a(len_1d)
    integer(c_int64_t), intent(out) :: out_index
    real(c_double), intent(out) :: out_value

    integer(c_int64_t) :: n, lo, hi, i, k, idx, t, nb1, j, fb, k0
    real(c_double) :: m, lm, cm, tailm
    integer :: nt, pos
    real(c_double) :: tmax(512)
    integer(c_int64_t) :: tidx(512)

    n = len_1d

    if (n == 0) then
      out_value = 0.0d0
      out_index = 0
      return
    end if

    if (n == 1) then
      out_value = a(1)
      out_index = 0
      return
    end if

    ! small: plain serial scan
    if (n < 8192) then
      m = a(1)
      idx = 1
      do i = 2, n
        if (a(i) > m) then
          m = a(i)
          idx = i
        end if
      end do
      out_value = m
      out_index = idx - 1
      return
    end if

    nt = omp_get_max_threads()
    if (nt > 512) nt = 512
    if (nt < 1) nt = 1
    if (int(n / 8192) < nt) nt = max(1, int(n / 8192))

    if (n / 8 + 1 > bmax_cap) then
      if (allocated(bmax)) deallocate(bmax)
      bmax_cap = n / 8 + 1
      allocate(bmax(bmax_cap))
    end if

    tidx = 0

    !$omp parallel default(none) shared(a, n, nt, tmax, tidx, bmax) private(t, lo, hi, i, k, k0, j, nb1, fb, m, lm, cm, tailm, idx, pos)
      t = omp_get_thread_num() + 1
      lo = ((n - 1) * (t - 1) / nt) / 8 * 8 + 1
      hi = ((n - 1) * t / nt) / 8 * 8
      if (t == nt) hi = n
      hi = min(hi, n)

      if (lo > hi) then
        tidx(t) = 0
      else
        j = (lo - 1) / 8 + 1
        nb1 = (hi - lo) / 8 + 1
        do k = lo, hi - 7, 8
          cm = max(max(a(k), a(k + 1)), max(a(k + 2), a(k + 3)))
          cm = max(cm, max(a(k + 4), a(k + 5)))
          cm = max(cm, max(a(k + 6), a(k + 7)))
          bmax(j) = cm
          j = j + 1
        end do
        ! tail (0..7 elements)
        if (k <= hi) then
          tailm = a(k)
          do i = k + 1, hi
            tailm = max(tailm, a(i))
          end do
          bmax(j) = tailm
          nb1 = j - (lo - 1) / 8
        end if

        cm = maxval(bmax((lo - 1) / 8 + 1 : (lo - 1) / 8 + nb1))
        if (cm == cm) then
          fb = findloc(bmax((lo - 1) / 8 + 1 : (lo - 1) / 8 + nb1), cm, dim=1)
          k0 = lo + 8 * (fb - 1)
          idx = k0
          do i = k0, min(k0 + 7, hi)
            if (a(i) == cm) then
              idx = i
              exit
            end if
          end do
          lm = cm
        else
          ! chunk contains NaN: exact serial fallback
          lm = a(lo)
          idx = lo
          do i = lo + 1, hi
            if (a(i) > lm) then
              lm = a(i)
              idx = i
            end if
          end do
        end if
        tmax(t) = lm
        tidx(t) = idx
      end if
    !$omp end parallel

    m = tmax(1)
    idx = tidx(1)
    do t = 2, nt
      if (tidx(t) > 0) then
        if (tmax(t) > m) then
          m = tmax(t)
          idx = tidx(t)
        else if (tmax(t) == m .and. tidx(t) < idx) then
          idx = tidx(t)
        end if
      end if
    end do

    out_value = m
    out_index = idx - 1
  end subroutine
end module amod
