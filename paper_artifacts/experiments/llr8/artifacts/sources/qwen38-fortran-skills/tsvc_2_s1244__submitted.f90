subroutine tsvc_2_s1244_fp64(a, b, c, d, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(inout) :: d(len_1d)
  real(c_double), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: n, nb, bs, k, lo, hi, i
  integer :: nt
  real(c_double) :: t, u
  u = 0.0d0

  n = len_1d
  if (n < 2) return

  if (n < 65536) then
    ! serial exact path (small n: avoid fork overhead)
    do i = 1, n - 1
      t = a(i + 1)
      a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
      d(i) = a(i) + t
    end do
    return
  end if

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  if (nt > 512) nt = 512
  nb = nt
  bs = (n - 1 + nb - 1) / nb

  ! Phase 1: every thread captures the FIRST element its right neighbor will
  ! clobber (a(hi+1)) before any write happens. The only cross-thread touch of
  ! that element is this read.
  !$omp parallel private(k, lo, hi, t, u)
  !$omp do schedule(static)
  do k = 0, nb - 1
    lo = k * bs + 1
    hi = min(lo + bs - 1, n - 1)
    if (lo <= hi) u = a(hi + 1)
  end do
  ! implicit barrier: all boundary reads done before any write below
  !$omp do schedule(static)
  do k = 0, nb - 1
    lo = k * bs + 1
    hi = min(lo + bs - 1, n - 1)
    if (lo <= hi) then
      do i = lo, hi - 1
        t = a(i + 1)
        a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
        d(i) = a(i) + t
      end do
      a(hi) = b(hi) + c(hi) * c(hi) + b(hi) * b(hi) + c(hi)
      d(hi) = a(hi) + u
    end if
  end do
  !$omp end parallel
end subroutine tsvc_2_s1244_fp64
