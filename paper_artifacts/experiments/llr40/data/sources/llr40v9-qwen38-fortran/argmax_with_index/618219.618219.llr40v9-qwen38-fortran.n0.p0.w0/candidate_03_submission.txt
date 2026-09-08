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
  integer :: nt, i
  integer(8) :: n, u, g, lo, hi
  real(8) :: x, v, v1, v2, v3, v4
  integer(8) :: ip2

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

  if (n < 16384_8) then
    x = ap(1)
    ip = 1
    g = (n - 2) / 4
    do u = 2, 4 * g - 2, 4
      v1 = ap(u)
      if (v1 > x) then
        x = v1
        ip = u
      end if
      v2 = ap(u + 1)
      if (v2 > x) then
        x = v2
        ip = u + 1
      end if
      v3 = ap(u + 2)
      if (v3 > x) then
        x = v3
        ip = u + 2
      end if
      v4 = ap(u + 3)
      if (v4 > x) then
        x = v4
        ip = u + 3
      end if
    end do
    u = 2 + 4 * g
    do while (u <= n)
      v = ap(u)
      if (v > x) then
        x = v
        ip = u
      end if
      u = u + 1
    end do
    vp = x
    return
  end if

  !$omp parallel default(none) shared(ap, tx, ti, n, nt) private(i, u, g, lo, hi, x, v, v1, v2, v3, v4, ip2)
  nt = omp_get_num_threads()
  i = omp_get_thread_num() + 1
  if (i <= nt) then
    lo = (n * (i - 1)) / nt + 1
    hi = (n * i) / nt
    if (lo <= hi) then
      x = ap(lo)
      ip2 = lo
      g = (hi - lo - 1) / 4
      do u = lo + 1, lo + 4 * g, 4
        v1 = ap(u)
        if (v1 > x) then
          x = v1
          ip2 = u
        end if
        v2 = ap(u + 1)
        if (v2 > x) then
          x = v2
          ip2 = u + 1
        end if
        v3 = ap(u + 2)
        if (v3 > x) then
          x = v3
          ip2 = u + 2
        end if
        v4 = ap(u + 3)
        if (v4 > x) then
          x = v4
          ip2 = u + 3
        end if
      end do
      u = lo + 1 + 4 * g
      do while (u <= hi)
        v = ap(u)
        if (v > x) then
          x = v
          ip2 = u
        end if
        u = u + 1
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
