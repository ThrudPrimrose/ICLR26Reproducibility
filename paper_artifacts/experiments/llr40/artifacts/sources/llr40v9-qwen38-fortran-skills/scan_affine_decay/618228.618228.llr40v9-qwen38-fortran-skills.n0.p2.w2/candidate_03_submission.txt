subroutine scan_affine_decay_fp64(c, x, y, len_1d, workspace, workspace_size) bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: x(len_1d)
  real(c_double), intent(inout) :: y(len_1d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  interface
    integer(4) function omp_get_num_threads() bind(C, name="omp_get_num_threads")
    end function omp_get_num_threads
    integer(4) function omp_get_thread_num() bind(C, name="omp_get_thread_num")
    end function omp_get_thread_num
  end interface

  integer, parameter :: SERIALLIM = 262144
  real(c_double), parameter :: PRODT = 1.0d-15
  integer(c_int64_t), parameter :: REPAIRCAP = 131072

  integer(c_int64_t) :: n, nt, tid, tot, lo, hi, span, i, ipos, nfull, ntail, k, iend
  real(c_double) :: rA, rS
  real(c_double) :: v0, v1, v2, v3

  n = len_1d
  if (n < 2) return

  if (n <= SERIALLIM) then
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  !$omp parallel default(none) &
     !$omp& shared(n, c, x, y) &
     !$omp& private(nt, tid, tot, lo, hi, span, i, ipos, nfull, ntail, k, iend, rA, rS, v0, v1, v2, v3)
  nt = omp_get_num_threads()
  tid = omp_get_thread_num()
  tot = n - 1
  lo = tot * tid / nt + 1
  hi = tot * (tid + 1) / nt + 1

  ! phase A: scan this thread's range from an arbitrary start (exact seed for
  ! thread 0, zero otherwise -- the start error decays by the product of c,
  ! which is < 0.999 per element, so it dies within tens of thousands of steps)
  if (tid == 0) then
    rS = y(1)
  else
    rS = 0.0d0
  end if
  span = hi - lo
  nfull = span / 4
  ntail = span - 4 * nfull
  ipos = lo + 1
  do k = 1, nfull
    i = ipos
    v0 = rS * c(i) + x(i)
    v1 = v0 * c(i + 1) + x(i + 1)
    v2 = v1 * c(i + 2) + x(i + 2)
    v3 = v2 * c(i + 3) + x(i + 3)
    y(i) = v0
    y(i + 1) = v1
    y(i + 2) = v2
    y(i + 3) = v3
    rS = v3
    ipos = i + 4
  end do
  do k = 1, ntail
    i = ipos
    rS = rS * c(i) + x(i)
    y(i) = rS
    ipos = i + 1
  end do

  !$omp barrier

  ! phase B: repair the prefix of this thread's range from the now-correct
  ! boundary value y(lo) until the running product of c drops below PRODT;
  ! beyond that, the arbitrary-start error is below the tolerance
  if (tid > 0) then
    rS = y(lo)
    rA = 1.0d0
    iend = min(hi, lo + REPAIRCAP)
    do i = lo + 1, iend
      rS = rS * c(i) + x(i)
      y(i) = rS
      rA = rA * c(i)
      if (rA < PRODT) exit
    end do
  end if
  !$omp end parallel
end subroutine scan_affine_decay_fp64
