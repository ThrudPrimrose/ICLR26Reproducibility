subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(c, name='tsvc_2_s2233_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: aa, bb, cc
  integer(c_int64_t), value :: len_2d

  integer(c_int64_t), parameter :: W64 = 16
  real(c_double), pointer :: a(:), b(:), cptr(:)
  integer(c_int64_t) :: n, n2, n2s, p, nrows, ngroups, rem, ntasks, j
  integer :: t, k, c0, it, tmax
  real(c_double) :: acc(16)
  real(c_double) :: v

  n = len_2d
  if (n <= 8) return
  n2 = n * n
  n2s = min(n2, 2147483646_8)
  call c_f_pointer(aa, a, [int(n2s)])
  call c_f_pointer(bb, b, [int(n2s)])
  call c_f_pointer(cc, cptr, [int(n2s)])
  if (n > 46000) then
    do j = 8, n-1
      do k = 8, int(n-1)
        a(j*n + k) = a((j-1)*n + k) + cptr(j*n + k)
      end do
    end do
    do j = 8, n-1
      do k = 8, int(n-1)
        b(j*n + k) = b((j-1)*n + k) + cptr(j*n + k)
      end do
    end do
    return
  end if
  nrows = n - 8
  ngroups = nrows / W64
  rem = mod(nrows, W64)
  ntasks = 2*ngroups + 2*rem
  tmax = int(ntasks) - 1

!$omp parallel do schedule(static) private(c0, it, v, j, k, p, acc)
  do t = 0, tmax
    if (t < int(ngroups)) then
      c0 = 8 + t*16
      do k = 1, 16
        acc(k) = a(7*n + c0 + k)
      end do
      p = 8*n + c0 + 1
      do j = 8, n-1
        do k = 1, 16
          acc(k) = acc(k) + cptr(p + k - 1)
        end do
        do k = 1, 16
          a(p + k - 1) = acc(k)
        end do
        p = p + n
      end do
    else if (t < 2*int(ngroups)) then
      c0 = 8 + (t - int(ngroups))*16
      do k = 1, 16
        acc(k) = b(7*n + c0 + k)
      end do
      p = 8*n + c0 + 1
      do j = 8, n-1
        do k = 1, 16
          acc(k) = acc(k) + cptr(p + k - 1)
        end do
        do k = 1, 16
          b(p + k - 1) = acc(k)
        end do
        p = p + n
      end do
    else if (t < 2*int(ngroups) + int(rem)) then
      it = 8 + int(ngroups*W64) + (t - 2*int(ngroups))
      v = a(7*n + it + 1)
      p = 8*n + it + 1
      do j = 8, n-1
        v = v + cptr(p)
        a(p) = v
        p = p + n
      end do
    else
      it = 8 + int(ngroups*W64) + (t - 2*int(ngroups) - int(rem))
      v = b(7*n + it + 1)
      p = 8*n + it + 1
      do j = 8, n-1
        v = v + cptr(p)
        b(p) = v
        p = p + n
      end do
    end if
  end do
end subroutine
