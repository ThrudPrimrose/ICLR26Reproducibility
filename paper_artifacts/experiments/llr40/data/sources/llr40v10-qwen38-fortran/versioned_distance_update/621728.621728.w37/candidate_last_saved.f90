! versioned_distance_update: a[i] = 0.75*a[i-K] + b[i]*c[i], i = K..N-1 (0-based).
! ABI (probed): subroutine(a*, b*, c*, int64 K, int64 LEN) all by value.
!
! Row view: row m = a[m*K .. m*K+K-1]; row m = 0.75*row m-1 + b_row*c_row,
! row 0 = seed. With the 0.75 decay a row depends on earlier rows only via
! the raw products d=b*c weighted by 0.75^j, j=1..256 (0.75^256 ~ 1.3e-31),
! so a slice of G consecutive rows is independent of other slices -> parallel.
subroutine versioned_distance_update_fp64(a, b, c, k, n) bind(c, name='versioned_distance_update_fp64')
  use iso_c_binding
  implicit none
  real(c_double) :: a(*)
  real(c_double) :: b(*)
  real(c_double) :: c(*)
  integer(c_int64_t), value :: k
  integer(c_int64_t), value :: n

  interface
    integer function omp_get_max_threads() bind(c, name='omp_get_max_threads')
    end function
  end interface

  integer(c_int64_t) :: i, m, r, j
  integer(c_int64_t) :: n_rows, m0, m1, mseed, g, nslices, p, rem, base
  real(c_double) :: w(256)
  real(c_double) :: t, carry
  real(c_double), pointer :: bufa(:), bufb(:)
  real(c_double), allocatable, target :: ta(:), tb(:)
  real(c_double), pointer :: tswap(:)
  integer :: p_int

  if (k < 1) return
  n_rows = (n - 1) / k
  if (n_rows < 1) return

  ! weights w(j) = 0.75**j
  t = 1.0d0
  do j = 1, 256
     t = t * 0.75d0
     w(j) = t
  end do

  p_int = omp_get_max_threads()
  if (p_int < 1) p_int = 1
  p = int(p_int, 8)
  ! ~4 slices per thread; keep a slice's working set below ~24 MB
  g = max(int(1_8), n_rows / (4 * p))
  g = min(g, max(int(256_8), int(24_8 * 1024 * 1024 / (8 * k))))
  g = min(g, n_rows)
  nslices = (n_rows + g - 1) / g
  rem = n - n_rows * k          ! length of the final (possibly short) row

  if (k == 1) then
     ! single chain: pure scalar FMA chain with a 256-product decay prologue
     !$omp parallel do schedule(static) private(m0, m1, mseed, carry)
     do i = 0, nslices - 1
        m0 = i * g + 1
        m1 = min(m0 + g - 1, n_rows)
        mseed = m0 - 1
        if (mseed == 0) then
           carry = a(1)
        else if (mseed <= 256) then
           carry = w(mseed) * a(1)
           do j = 1, mseed - 1
              carry = carry + w(j) * b(mseed - j + 1) * c(mseed - j + 1)
           end do
        else
           carry = 0.0d0
           do j = 1, 256
              carry = carry + w(j) * b(mseed - j + 1) * c(mseed - j + 1)
           end do
        end if
        do m = m0, m1
           carry = 0.75d0 * carry + b(m + 1) * c(m + 1)
           a(m + 1) = carry
        end do
     end do
  else
     !$omp parallel do schedule(static) private(bufa, bufb, ta, tb, tswap, m0, m1, mseed, base, carry)
     do i = 0, nslices - 1
        m0 = i * g + 1
        m1 = min(m0 + g - 1, n_rows)
        mseed = m0 - 1
        allocate(ta(k), tb(k))
        bufa => ta
        bufb => tb
        if (mseed == 0) then
           do r = 1, k
              bufa(r) = a(r)
           end do
        else if (mseed <= 256) then
           carry = w(mseed)
           do r = 1, k
              bufa(r) = carry * a(r)
           end do
           do j = 1, mseed - 1
              base = (mseed - j) * k
              do r = 1, k
                 bufa(r) = bufa(r) + w(j) * b(base + r) * c(base + r)
              end do
           end do
        else
           do r = 1, k
              bufa(r) = 0.0d0
           end do
           do j = 1, 256
              base = (mseed - j) * k
              do r = 1, k
                 bufa(r) = bufa(r) + w(j) * b(base + r) * c(base + r)
              end do
           end do
        end if
        do m = m0, m1
           base = m * k
           if (m < n_rows) then
              do r = 1, k
                 bufb(r) = 0.75d0 * bufa(r) + b(base + r) * c(base + r)
                 a(base + r) = bufb(r)
              end do
           else
              do r = 1, rem
                 bufb(r) = 0.75d0 * bufa(r) + b(base + r) * c(base + r)
                 a(base + r) = bufb(r)
              end do
           end if
           tswap => bufa
           bufa => bufb
           bufb => tswap
        end do
        deallocate(ta, tb)
     end do
  end if
end subroutine
