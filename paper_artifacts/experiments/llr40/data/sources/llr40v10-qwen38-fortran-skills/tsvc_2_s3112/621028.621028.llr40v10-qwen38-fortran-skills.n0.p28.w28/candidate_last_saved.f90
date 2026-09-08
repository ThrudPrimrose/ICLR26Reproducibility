subroutine tsvc_2_s3112_fp64(a, b, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in)    :: a(len_1d)
  real(c_double), intent(out)   :: b(len_1d)

  integer, parameter :: MAXT = 256
  integer :: nt, c
  integer(c_int64_t) :: n, base, rem, i, lo(MAXT), hi(MAXT)
  real(c_double) :: part(0:MAXT), s, run

  n = len_1d
  if (n <= 0) return
  nt = omp_get_max_threads()
  if (nt > MAXT) nt = MAXT
  if (nt < 1) nt = 1
  if (nt > int(n)) nt = int(n)
  base = n / int(nt, 8)
  rem = mod(n, int(nt, 8))
  do c = 1, nt
    lo(c) = int(c - 1, 8) * base + min(int(c - 1, 8), rem) + 1
    hi(c) = int(c, 8) * base + min(int(c, 8), rem)
  end do

  ! Pass 1: per-chunk sums
  !$omp parallel do
  do c = 1, nt
    s = 0.0d0
    !$omp simd reduction(+:s)
    do i = lo(c), hi(c)
      s = s + a(i)
    end do
    part(c) = s
  end do

  ! Prefix over the chunk sums (serial, tiny)
  do c = 2, nt
    part(c) = part(c) + part(c - 1)
  end do

  ! Pass 2: in-chunk inclusive scan, starting from the exclusive chunk offset
  part(0) = 0.0d0
  !$omp parallel do
  do c = 1, nt
    run = part(c - 1)
    !$omp simd reduction(inscan,+:run)
    do i = lo(c), hi(c)
      run = run + a(i)
      !$omp scan inclusive(run)
      b(i) = run
    end do
  end do
end subroutine tsvc_2_s3112_fp64
