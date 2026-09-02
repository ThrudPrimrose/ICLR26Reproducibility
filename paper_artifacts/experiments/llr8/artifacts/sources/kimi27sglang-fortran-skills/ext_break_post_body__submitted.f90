subroutine ext_break_post_body_fp64(a, b, c, n, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n), c(n)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, k, m, lo, hi, lmin
  integer(c_int) :: nt, t
  integer(c_int64_t), allocatable :: local_min(:)
  if (n <= 0) return
  nt = omp_get_max_threads()
  allocate(local_min(nt))
  local_min = n + 1
  !$omp parallel private(t, lo, hi, i, lmin)
  t = omp_get_thread_num()
  lo = (n * int(t, c_int64_t)) / int(nt, c_int64_t) + 1
  hi = (n * int(t + 1, c_int64_t)) / int(nt, c_int64_t)
  lmin = n + 1
  !$omp simd reduction(min:lmin)
  do i = lo, hi
    lmin = min(lmin, merge(i, n + 1, c(i) > b(i)))
  end do
  local_min(t + 1) = lmin
  !$omp end parallel
  k = minval(local_min(1:nt))
  deallocate(local_min)
  m = k
  if (m > n) m = n
  !$omp parallel private(t, lo, hi, i)
  t = omp_get_thread_num()
  lo = (n * int(t, c_int64_t)) / int(nt, c_int64_t) + 1
  hi = (n * int(t + 1, c_int64_t)) / int(nt, c_int64_t)
  if (lo <= m) then
    if (hi > m) hi = m
    !$omp simd
    do i = lo, hi
      a(i) = a(i) + b(i) * c(i)
    end do
  end if
  !$omp end parallel
end subroutine ext_break_post_body_fp64
