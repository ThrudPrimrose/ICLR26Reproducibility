subroutine tsvc_2_s231_fp64(aa, bb, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int64_t) :: j, lo, hi
  integer :: t, nt, maxt
  real(8) :: t0, t1
  if (len_2d <= 1) return
  maxt = omp_get_max_threads()
  t0 = omp_get_wtime()
  !$omp parallel shared(aa, bb, len_2d) private(t, nt, lo, hi, j)
  nt = omp_get_num_threads()
  t = omp_get_thread_num()
  lo = (len_2d * t) / nt + 1
  hi = (len_2d * (t + 1)) / nt
  if (lo <= hi) then
    do j = 2, len_2d
      aa(lo:hi, j) = aa(lo:hi, j - 1) + bb(lo:hi, j)
    end do
  end if
  !$omp end parallel
  t1 = omp_get_wtime()
  write(*,'(A,F12.4,A,I4,A,F12.4)') 'KERNEL_MS=', real((t1-t0)*1e3,8), ' maxt=', maxt, ' GBps(3L2)=', 3.0d0*len_2d*len_2d*8/((t1-t0))/1e9
  flush(0)
end subroutine tsvc_2_s231_fp64
