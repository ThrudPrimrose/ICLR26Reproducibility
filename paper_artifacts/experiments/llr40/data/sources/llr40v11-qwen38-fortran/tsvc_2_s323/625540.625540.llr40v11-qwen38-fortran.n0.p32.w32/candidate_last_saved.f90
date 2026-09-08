subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C, name='tsvc_2_s323_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  real(c_double), intent(inout), dimension(*) :: a, b
  real(c_double), intent(in),    dimension(*) :: c, d, e
  integer(c_int64_t), value, intent(in)       :: len_1d
  integer(c_int64_t) :: i, n
  real(c_double) :: s, w0, w1
  n = len_1d
  w0 = omp_get_wtime()
  !$omp parallel do schedule(static)
  do i = 2_c_int64_t, n
    a(i) = 1.0d0; b(i) = 1.0d0
  end do
  w1 = omp_get_wtime()
  write(*,'(A,ES11.3,A,ES11.3)') 'PREFLT_PAR t_ms=', 1.0d3*(w1-w0), ' ns/elt=', 1.0d9*(w1-w0)/n
  w0 = omp_get_wtime()
  do i = 2_c_int64_t, n
    a(i) = 2.0d0; b(i) = 2.0d0
  end do
  w1 = omp_get_wtime()
  write(*,'(A,ES11.3,A,ES11.3)') 'WRITEWARM_S t_ms=', 1.0d3*(w1-w0), ' ns/elt=', 1.0d9*(w1-w0)/n
  w0 = omp_get_wtime()
  do i = 2_c_int64_t, n
    a(i) = b(i-1) + c(i)*d(i)
    b(i) = a(i) + c(i)*e(i)
  end do
  w1 = omp_get_wtime()
  write(*,'(A,ES11.3,A,ES11.3,A,ES12.4)') 'KERNEL_S t_ms=', 1.0d3*(w1-w0), ' ns/elt=', 1.0d9*(w1-w0)/n, ' a2=', a(2)
  s = 0.0d0
  w0 = omp_get_wtime()
  !$omp parallel do schedule(static) reduction(+:s)
  do i = 2_c_int64_t, n
    s = s + c(i) + d(i) + e(i)
  end do
  w1 = omp_get_wtime()
  write(*,'(A,ES11.3,A,ES11.3,A,ES12.4)') 'P1_SCAN_PAR t_ms=', 1.0d3*(w1-w0), ' ns/elt=', 1.0d9*(w1-w0)/n, ' s=', s
  flush(6)
end subroutine tsvc_2_s323_fp64
