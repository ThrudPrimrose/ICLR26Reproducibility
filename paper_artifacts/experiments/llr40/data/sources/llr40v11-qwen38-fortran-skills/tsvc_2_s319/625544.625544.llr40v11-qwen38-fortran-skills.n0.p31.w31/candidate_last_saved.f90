subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  double precision :: sum_val, t0, t1
  integer(c_int64_t) :: i
  integer :: unit
  double precision :: ms_ans, ms_nosum, ms_copy
  ! measurement 1: no reduction (same 5 memory ops)
  t0 = omp_get_wtime()
  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = c(i) + d(i)
    b(i) = c(i) + e(i)
  end do
  t1 = omp_get_wtime(); ms_nosum = (t1-t0)*1000.0d0
  ! measurement 2: pure copy (2 ops)
  t0 = omp_get_wtime()
  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = c(i)
  end do
  t1 = omp_get_wtime(); ms_copy = (t1-t0)*1000.0d0
  ! correct answer LAST
  t0 = omp_get_wtime()
  sum_val = 0.0d0
  !$omp parallel do simd reduction(+:sum_val)
  do i = 1, len_1d
    a(i) = c(i) + d(i)
    sum_val = sum_val + a(i)
    b(i) = c(i) + e(i)
    sum_val = sum_val + b(i)
  end do
  b(1) = sum_val
  t1 = omp_get_wtime(); ms_ans = (t1-t0)*1000.0d0
  open(newunit=unit, file='/shared/agent-31/diag_s319.txt', status='replace', action='write')
  write(unit, '(A,I0,A,ES11.3,A,ES11.3,A,ES11.3)') 'n=', len_1d, ' ans_ms=', ms_ans, ' nosum_ms=', ms_nosum, ' copy_ms=', ms_copy
  close(unit)
end subroutine tsvc_2_s319_fp64
