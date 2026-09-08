subroutine tsvc_2_s3110_fp64(aa, bb, len2d) bind(C, name='tsvc_2_s3110_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), dimension(*), intent(in)  :: aa
  real(c_double), dimension(*), intent(out) :: bb
  integer(c_int64_t), value, intent(in) :: len2d

  real(c_double), dimension(:), allocatable :: lm
  integer(c_int64_t), dimension(:), allocatable :: lp
  integer(c_int64_t) :: n, t0, t1, per, pos, xi, yi, abits

interface
  subroutine scan_region(a, best, pos, n)
    real(8), dimension(:), intent(in) :: a
    real(8), intent(inout) :: best
    integer(8), intent(inout) :: pos
    integer(8), intent(in) :: n
  end subroutine scan_region
end interface

  real(c_double) :: best, bestv
  integer :: tid, nt

  if (len2d <= 0_8) return
  n = len2d * len2d

  ! If aa(1) is NaN every strict comparison in the reference fails, so the
  ! result is NaN with indices (0,0).  Bit pattern test: exponent field all
  ! ones and non-zero significand.
  abits = transfer(aa(1), 0_8)
  if (iand(ishft(abits, -52), 2047_8) == 2047_8 .and. &
       iand(abits, 4503599627370495_8) /= 0_8) then
    bb(1) = aa(1)
    return
  end if

  nt = omp_get_max_threads()
  allocate(lm(nt), lp(nt))

!$omp parallel private(tid,t0,t1,per,best,pos)
  tid = omp_get_thread_num()
  per = (n + int(omp_get_num_threads(), 8) - 1_8) / int(omp_get_num_threads(), 8)
  t0 = int(tid, 8) * per
  t1 = min(t0 + per, n)
  best = -huge(1.0d0)
  pos = -1_8
  call scan_region(aa(t0+1:t1), best, pos, t1 - t0)
  pos = pos + t0
  lm(tid+1) = best
  lp(tid+1) = pos
!$omp end parallel

  bestv = aa(1)
  pos = 0_8
  do tid = 1, nt
    if (lm(tid) > bestv) then
      bestv = lm(tid)
      pos = lp(tid)
    end if
  end do
  xi = pos / len2d
  yi = pos - xi*len2d
  bb(1) = bestv + real(xi, 8) + real(yi, 8)
  deallocate(lm, lp)
end subroutine

! Scans a(1:n) in row-major element order for the first maximum (strict >).
! On return best is the maximum value (or -inf if n == 0) and pos the 0-based
! position of its first occurrence inside the region.
subroutine scan_region(a, best, pos, n)
  implicit none
  real(8), dimension(:), intent(in) :: a
  real(8), intent(inout) :: best
  integer(8), intent(inout) :: pos
  integer(8), intent(in) :: n
  integer(8) :: c, wbase, lo, sub, half, m
  integer, parameter :: W = 1024

  do c = 0, n/W - 1_8
    wbase = c * int(W, 8)
    if (any(a(wbase+1 : wbase+W) > best)) then
      lo = wbase
      sub = int(W, 8)
      do while (sub > 1_8)
        half = sub / 2_8
        if (any(a(lo+1 : lo+half) > best)) then
          sub = half
        else
          lo = lo + half
          sub = half
        end if
      end do
      best = a(lo+1)
      pos = lo
    end if
  end do
  do m = (n/W) * W, n - 1_8
    if (a(m+1) > best) then
      best = a(m+1)
      pos = m
    end if
  end do
end subroutine
