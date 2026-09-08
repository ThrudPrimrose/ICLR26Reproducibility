subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d), b(len_1d)
  real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)
  real(c_double) :: part(0:63), TOT, TOTPRE, R, R1, sumt, tmp, b0
  integer(c_int64_t) :: i, lo, hi, lo2, hi2, n, nblk, bszl, nblki, blk, k, j
  integer :: nt, t

  n = len_1d
  if (n <= 1) return
  b0 = b(1)

  if (n < 400000_8) then
     do i = 2, n
        R1 = b(i - 1) + c(i) * d(i)
        a(i) = R1
        b(i) = R1 + c(i) * e(i)
     end do
     return
  end if

  nt = omp_get_max_threads()
  if (nt > 64) nt = 64
  blk = 1500000_8
  do
     if (n - 1 <= blk * nt) exit
     blk = blk * 2
  end do
  nblk = (n - 1 + blk - 1) / blk

  TOT = b0
  !$omp parallel num_threads(nt) private(t, lo, hi, lo2, hi2, i, R, R1, sumt, j, bszl, nblki) shared(part, TOT, TOTPRE)
  t = omp_get_thread_num()
  nblki = nblk
  bszl = blk
  do k = 0, nblki - 1
     lo = k * bszl + 2
     hi = min(lo + bszl - 1, n)
     bszl = hi - lo + 1
     lo2 = lo + bszl * t / nt
     hi2 = lo + bszl * (t + 1) / nt - 1
     if (lo2 <= hi2) then
        sumt = 0.0d0
        !$omp simd reduction(+:sumt)
        do i = lo2, hi2
           sumt = sumt + c(i) * (d(i) + e(i))
        end do
        part(t) = sumt
     else
        part(t) = 0.0d0
     end if
     !$omp barrier
     if (t == 0) then
        TOTPRE = TOT
        tmp = 0.0d0
        do j = 0, nt - 1
           sumt = part(j)
           part(j) = tmp
           tmp = tmp + sumt
        end do
        TOT = TOTPRE + tmp
     end if
     !$omp barrier
     if (lo2 <= hi2) then
        R = TOTPRE + part(t)
        do i = lo2, hi2
           R1 = R + c(i) * d(i)
           a(i) = R1
           R = R1 + c(i) * e(i)
           b(i) = R
        end do
     end if
  end do
  !$omp end parallel
end subroutine tsvc_2_s323_fp64
