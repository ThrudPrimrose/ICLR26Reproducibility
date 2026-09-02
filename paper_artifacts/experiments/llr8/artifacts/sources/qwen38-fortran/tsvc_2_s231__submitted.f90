! TSVC s231: aa[j,i] = aa[j-1,i] + bb[j,i]  (C row-major flat, 0-based)
! j (row) direction carries the true dependence -> kept serial (2-way unrolled).
! i (column) direction is independent and unit stride -> vectorized.
! Each OpenMP thread owns ONE contiguous column range of ~n/nt columns.
subroutine tsvc_2_s231_fp64(aa, bb, len_2d, ws, ws_bytes) bind(C, name="tsvc_2_s231_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_ptr, c_associated
  use omp_lib, only: omp_get_max_threads
  implicit none
  real(c_double), intent(inout), dimension(*) :: aa
  real(c_double), intent(in),    dimension(*) :: bb
  integer(c_int64_t), value, intent(in) :: len_2d
  type(c_ptr), intent(in) :: ws
  integer(c_int64_t), value, intent(in) :: ws_bytes
  integer(c_int64_t) :: i, j, n, b, c0, c1, nb, bsize, jmax
  integer(c_int64_t) :: nt
  n = len_2d
  if (c_associated(ws) .and. ws_bytes > 0_c_int64_t) then
    ! scratch provided; this kernel needs none
  end if
  nt = int(omp_get_max_threads(), 8_c_int64_t)
  nb = n
  if (nb > nt) nb = nt
  bsize = (n + nb - 1) / nb
  bsize = (bsize + 7) / 8 * 8
  nb = (n + bsize - 1) / bsize
!$omp parallel do
  do b = 1, nb
    c0 = (b - 1) * bsize + 1
    c1 = min(n, b * bsize)
    jmax = n - 1 - mod(n - 1, 2_c_int64_t)
    do j = 2, jmax, 2
      do i = c0, c1
        aa((j-1)*n + i) = aa((j-2)*n + i) + bb((j-1)*n + i)
      end do
      do i = c0, c1
        aa(j*n + i) = aa((j-1)*n + i) + bb(j*n + i)
      end do
    end do
    if (mod(n - 1, 2_c_int64_t) == 1) then
      do i = c0, c1
        aa((n-1)*n + i) = aa((n-2)*n + i) + bb((n-1)*n + i)
      end do
    end if
  end do
end subroutine tsvc_2_s231_fp64
