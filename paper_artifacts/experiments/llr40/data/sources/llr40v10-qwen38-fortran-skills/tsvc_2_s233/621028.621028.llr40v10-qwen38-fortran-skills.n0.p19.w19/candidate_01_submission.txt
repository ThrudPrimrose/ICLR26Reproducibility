subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)
  integer(kind=8) :: n, i, j, nt, m, block, nb, t, lo, hi
  real(c_double) :: s

  n = len_2d
  if (n <= 8) return

  ! Loop A: aa(i,j) = aa(i,j-1) + cc(i,j); chain along j (contiguous),
  ! parallel across columns i. Chain runs in a register.
  !$omp parallel do
  do i = 9, n
    s = aa(i, 8)
    do j = 9, n
      s = s + cc(i, j)
      aa(i, j) = s
    end do
  end do

  ! Loop B: bb(i,j) = bb(i-1,j) + cc(i,j); chain along i (contiguous),
  ! parallel across rows: each thread owns a band of rows and walks i.
  nt = omp_get_max_threads()
  m = n - 8
  block = m / (2*nt)
  if (block < 16) block = 16
  nb = (m + block - 1) / block
  !$omp parallel do
  do t = 0, nb-1
    lo = 9 + t*block
    hi = min(n, lo + block - 1)
    do i = 9, n
      bb(i, lo:hi) = bb(i-1, lo:hi) + cc(i, lo:hi)
    end do
  end do
end subroutine tsvc_2_s233_fp64
