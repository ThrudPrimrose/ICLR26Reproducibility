! tsvc_2 s3112: b = inclusive prefix sum of a  (b(i) = sum(a(1..i)))
!
! Strategy:
!   small n : single serial scan (no thread fork)
!   large n : chunked two-pass scan.
!             pass 1: parallel vectorized chunk sums (one simd reduction per chunk)
!             pass 2: per chunk, SIMD inclusive scan seeded with the chunk offset
!             (reduction(inscan,+:s) + scan inclusive), fully vectorized.
subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(len_1d)

  integer(c_int64_t) :: n, i, lo, hi, chunk, nch
  integer :: nt, t
  real(c_double), allocatable :: part(:)
  real(c_double) :: run, s

  n = len_1d
  if (n <= 0) return
  if (n == 1) then
    b(1) = a(1)
    return
  end if

  if (n < 262144_8) then
    run = 0.0d0
    do i = 1, n
      run = run + a(i)
      b(i) = run
    end do
    return
  end if

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  nch = int(8 * nt, 8)
  chunk = (n + nch - 1) / nch
  chunk = ((chunk + 255_8) / 256_8) * 256_8
  nch = (n + chunk - 1) / chunk
  allocate(part(nch))

!$omp parallel
!$omp do schedule(static)
  do t = 1, int(nch)
    lo = int(t - 1, 8) * chunk + 1
    hi = min(int(t, 8) * chunk, n)
    run = 0.0d0
!$omp simd reduction(+:run)
    do i = lo, hi
      run = run + a(i)
    end do
    part(t) = run
  end do
!$omp end do
!$omp single
  do t = 2, int(nch)
    part(t) = part(t) + part(t - 1)
  end do
!$omp end single
!$omp do schedule(static)
  do t = 1, int(nch)
    lo = int(t - 1, 8) * chunk + 1
    hi = min(int(t, 8) * chunk, n)
    if (t == 1) then
      s = 0.0d0
    else
      s = part(t - 1)
    end if
!$omp simd reduction(inscan, +:s)
    do i = lo, hi
      s = s + a(i)
!$omp scan inclusive(s)
      b(i) = s
    end do
  end do
!$omp end do
!$omp end parallel

  deallocate(part)
end subroutine tsvc_2_s3112_fp64
