subroutine ext_war_unit_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  type(c_ptr), value, intent(in) :: workspace

  real(c_double) :: t(8)
  real(c_double) :: g = 0.0d0
  integer(c_int64_t) :: n, nb, nt, tid, lo, hi, i0, k, i, jj

  n = len_1d
  if (n < 2) return
  nb = (n - 1) / 32
  nt = omp_get_max_threads()
  if (nt < 1) nt = 1

  !$omp parallel default(none) shared(a, b, n, nb, nt) private(tid, lo, hi, i0, k, i, jj, g, t)
    tid = omp_get_thread_num()
    lo = nb * tid / nt + 1
    hi = nb * (tid + 1) / nt
    if (hi >= lo) g = a(32 * hi + 1)
    !$omp barrier
    do k = lo, hi - 1
      i0 = 32 * (k - 1) + 1
      do jj = 0, 24, 8
        t(:) = a(i0+jj+1 : i0+jj+8)
        a(i0+jj : i0+jj+7) = t(:) + b(i0+jj : i0+jj+7)
      end do
    end do
    if (hi >= lo) then
      i0 = 32 * (hi - 1) + 1
      do jj = 0, 16, 8
        t(:) = a(i0+jj+1 : i0+jj+8)
        a(i0+jj : i0+jj+7) = t(:) + b(i0+jj : i0+jj+7)
      end do
      t(1 : 7) = a(i0 + 25 : i0 + 31)
      t(8) = g
      a(i0 + 24 : i0 + 31) = t(:) + b(i0 + 24 : i0 + 31)
    end if
    if (tid == nt - 1) then
      do i = 32 * nb + 1, n - 1
        a(i) = a(i + 1) + b(i)
      end do
    end if
  !$omp end parallel
end subroutine ext_war_unit_fp64
