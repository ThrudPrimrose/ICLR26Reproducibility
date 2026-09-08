subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(inout) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D)
  real(c_double), intent(in) :: e(LEN_1D)
  integer(c_int64_t) :: i, nt, t, lo, hi
  real(c_double) :: sum_val, local_sum
  real(c_double), allocatable :: parts(:)

  nt = omp_get_max_threads()
  allocate(parts(nt))
  parts = 0.0d0
  !$omp parallel private(t, lo, hi, i, local_sum)
  t = omp_get_thread_num()
  lo = (LEN_1D * t) / nt + 1
  hi = (LEN_1D * (t + 1)) / nt
  local_sum = 0.0d0
  do i = lo, hi
    a(i) = c(i) + d(i)
    b(i) = c(i) + e(i)
    local_sum = local_sum + c(i) + d(i) + c(i) + e(i)
  end do
  parts(t + 1) = local_sum
  !$omp end parallel
  sum_val = 0.0d0
  do i = 1, nt
    sum_val = sum_val + parts(i)
  end do
  b(1) = sum_val
  deallocate(parts)
end subroutine tsvc_2_s319_fp64
