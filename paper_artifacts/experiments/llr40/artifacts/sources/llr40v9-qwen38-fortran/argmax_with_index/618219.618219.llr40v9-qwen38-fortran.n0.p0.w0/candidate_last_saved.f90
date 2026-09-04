subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) &
    bind(C, name="argmax_with_index_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value :: a
  type(c_ptr), value :: out_index
  type(c_ptr), value :: out_value
  integer(8), value :: len_1d

  interface
    function omp_get_num_threads() result(ntmp) bind(C)
      integer :: ntmp
    end function omp_get_num_threads
    function omp_get_thread_num() result(ttmp) bind(C)
      integer :: ttmp
    end function omp_get_thread_num
  end interface

  real(8), pointer, contiguous :: ap(:)
  integer(8), pointer :: ip
  real(8), pointer :: vp
  real(8) :: tx(256)
  integer(8) :: ti(256)
  integer :: nt
  integer(8) :: n, u, w, g, g2, lo, hi, i
  real(8) :: x, v, v1, v2, v3, v4, xA, xB, xC, xD
  integer(8) :: iA, iB, iC, iD, ip2
  real(8), parameter :: NIN = transfer(-4503599627370496_8, 0.0d0)

  n = len_1d
  call c_f_pointer(a, ap, [n])
  call c_f_pointer(out_index, ip)
  call c_f_pointer(out_value, vp)

  if (n <= 1) then
    if (n == 1) then
      vp = ap(1)
      ip = 1
    end if
    return
  end if

  if (ap(1) /= ap(1)) then
    vp = ap(1)
    ip = 1
    return
  end if

  if (n < 16384_8) then
    x = NIN
    ip = 1
    g = (n - 1) / 64
    do u = 1, 64 * g - 63, 64
      xA = NIN
      iA = u
      xB = NIN
      iB = u + 1
      xC = NIN
      iC = u + 2
      xD = NIN
      iD = u + 3
      do w = u, u + 60, 4
        v1 = ap(w)
        if (v1 > xA) then
          xA = v1
          iA = w
        end if
        v2 = ap(w + 1)
        if (v2 > xB) then
          xB = v2
          iB = w + 1
        end if
        v3 = ap(w + 2)
        if (v3 > xC) then
          xC = v3
          iC = w + 2
        end if
        v4 = ap(w + 3)
        if (v4 > xD) then
          xD = v4
          iD = w + 3
        end if
      end do
      if (xB > xA) then
        xA = xB
        iA = iB
      else if (xA > xB) then
      else if (iB < iA) then
        iA = iB
      end if
      if (xC > xA) then
        xA = xC
        iA = iC
      else if (xA > xC) then
      else if (iC < iA) then
        iA = iC
      end if
      if (xD > xA) then
        xA = xD
        iA = iD
      else if (xA > xD) then
      else if (iD < iA) then
        iA = iD
      end if
      if (xA > x) then
        x = xA
        ip = iA
      end if
    end do
    u = 64 * g + 1
    g2 = (n - u) / 4
    do w = u, u + 4 * g2 - 3, 4
      v1 = ap(w)
      if (v1 > x) then
        x = v1
        ip = w
      end if
      v2 = ap(w + 1)
      if (v2 > x) then
        x = v2
        ip = w + 1
      end if
      v3 = ap(w + 2)
      if (v3 > x) then
        x = v3
        ip = w + 2
      end if
      v4 = ap(w + 3)
      if (v4 > x) then
        x = v4
        ip = w + 3
      end if
    end do
    w = u + 4 * g2
    do while (w <= n)
      v = ap(w)
      if (v > x) then
        x = v
        ip = w
      end if
      w = w + 1
    end do
    vp = x
    return
  end if

  !$omp parallel default(none) shared(ap, tx, ti, n, nt) private(i, u, w, g, g2, lo, hi, x, v, v1, v2, v3, v4, xA, xB, xC, xD, iA, iB, iC, iD, ip2)
  nt = omp_get_num_threads()
  i = omp_get_thread_num() + 1
  if (i <= nt) then
    lo = (n * (i - 1)) / nt + 1
    hi = (n * i) / nt
    if (lo <= hi) then
      x = NIN
      ip2 = lo
      g = (hi - lo) / 64
      do u = lo, lo + 64 * g - 64, 64
        xA = NIN
        iA = u
        xB = NIN
        iB = u + 1
        xC = NIN
        iC = u + 2
        xD = NIN
        iD = u + 3
        do w = u, u + 60, 4
          v1 = ap(w)
          if (v1 > xA) then
            xA = v1
            iA = w
          end if
          v2 = ap(w + 1)
          if (v2 > xB) then
            xB = v2
            iB = w + 1
          end if
          v3 = ap(w + 2)
          if (v3 > xC) then
            xC = v3
            iC = w + 2
          end if
          v4 = ap(w + 3)
          if (v4 > xD) then
            xD = v4
            iD = w + 3
          end if
        end do
        if (xB > xA) then
          xA = xB
          iA = iB
        else if (xA > xB) then
        else if (iB < iA) then
          iA = iB
        end if
        if (xC > xA) then
          xA = xC
          iA = iC
        else if (xA > xC) then
        else if (iC < iA) then
          iA = iC
        end if
        if (xD > xA) then
          xA = xD
          iA = iD
        else if (xA > xD) then
        else if (iD < iA) then
          iA = iD
        end if
        if (xA > x) then
          x = xA
          ip2 = iA
        end if
      end do
      u = lo + 64 * g + 1
      g2 = (hi - u) / 4
      do w = u, u + 4 * g2 - 3, 4
        v1 = ap(w)
        if (v1 > x) then
          x = v1
          ip2 = w
        end if
        v2 = ap(w + 1)
        if (v2 > x) then
          x = v2
          ip2 = w + 1
        end if
        v3 = ap(w + 2)
        if (v3 > x) then
          x = v3
          ip2 = w + 2
        end if
        v4 = ap(w + 3)
        if (v4 > x) then
          x = v4
          ip2 = w + 3
        end if
      end do
      w = u + 4 * g2
      do while (w <= hi)
        v = ap(w)
        if (v > x) then
          x = v
          ip2 = w
        end if
        w = w + 1
      end do
      tx(i) = x
      ti(i) = ip2
    else
      ti(i) = 0
    end if
  end if
  !$omp end parallel

  x = ap(1)
  ip = 1
  do i = 1, nt
    if (ti(i) > 0) then
      if (tx(i) > x) then
        x = tx(i)
        ip = ti(i)
      end if
    end if
  end do
  vp = x
end subroutine argmax_with_index_fp64
