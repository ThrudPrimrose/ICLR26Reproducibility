subroutine tsvc_2_s252_fp64(a, b, c, len_1d, workspace, workspace_size) &
    bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in)    :: b(len_1d)
  real(c_double), intent(in)    :: c(len_1d)
  real(c_double), intent(inout) :: workspace(*)
  integer, parameter :: TS = 65536
  integer :: i, n, t, lo, hi, ntiles, tile
  real(c_double) :: p(0:TS - 1), pv
  n = int(len_1d)
  if (n <= 0) return
  if (n <= 16384) then
    ! tiny: stay serial, avoid thread-team spawn
    !$omp simd
    do i = 1, n
      p(i - 1) = b(i) * c(i)
    end do
    a(1) = p(0)
    !$omp simd
    do i = 2, n
      a(i) = p(i - 1) + p(i - 2)
    end do
    return
  end if
  ntiles = (n + TS - 1) / TS
  if (n <= 262144) ntiles = min(n, 96)
  tile = (n + ntiles - 1) / ntiles
  !$omp parallel do private(t, lo, hi, pv, i, p)
  do t = 0, ntiles - 1
    lo = t * tile + 1
    hi = min(n, lo + tile - 1)
    !$omp simd
    do i = lo, hi
      p(i - lo) = b(i) * c(i)
    end do
    pv = 0.0d0
    if (lo > 1) pv = b(lo - 1) * c(lo - 1)
    a(lo) = p(0) + pv
    if (hi > lo) then
      !$omp simd
      do i = lo + 1, hi
        a(i) = p(i - lo) + p(i - lo - 1)
      end do
    end if
  end do
end subroutine tsvc_2_s252_fp64
