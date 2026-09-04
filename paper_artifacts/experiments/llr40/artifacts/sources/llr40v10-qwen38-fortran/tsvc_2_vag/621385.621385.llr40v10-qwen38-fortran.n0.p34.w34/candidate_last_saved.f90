subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name='tsvc_2_vag_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(kind=c_double) :: a(*)
  real(kind=c_double), intent(in) :: b(*)
  integer(kind=c_int32_t), intent(in) :: ip(*)
  integer(kind=c_int64_t), value :: len_1d
  integer(c_int64_t) :: i, M, rep
  real(kind=c_double) :: acc, t0, t1
  M = min(len_1d, 10000000_8)
  acc = 0.0d0
  t0 = omp_get_wtime()
  do rep = 1, 8
    do i = 1, M
      acc = acc + b(i)
    end do
  end do
  t1 = omp_get_wtime()
  write(*,*) 'seq8B_read GBps=', 8.0*M*8.0/(t1-t0)/1.0d9, ' ms_total=', (t1-t0)*1000.0
  flush(0)
  t0 = omp_get_wtime()
  do rep = 1, 8
    do i = 1, M
      acc = acc + b(ip(i))
    end do
  end do
  t1 = omp_get_wtime()
  write(*,*) 'rand8B_read GBps=', 8.0*M*8.0/(t1-t0)/1.0d9, ' ms_total=', (t1-t0)*1000.0
  flush(0)
  a(1) = acc
end subroutine tsvc_2_vag_fp64
