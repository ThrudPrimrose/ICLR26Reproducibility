subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(c, name="tsvc_2_s119_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(inout) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(in) :: bb(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: i, j, tid, nthreads, chunk, j_start, j_end

  !$omp parallel private(i, j, tid, nthreads, chunk, j_start, j_end)
  nthreads = omp_get_num_threads()
  tid = omp_get_thread_num()
  chunk = (LEN_2D + nthreads - 1) / nthreads
  j_start = tid * chunk
  j_end = min(LEN_2D - 1, j_start + chunk - 1)
  if (j_start == 0) j_start = 1

  do i = 1, LEN_2D-1
    do j = j_start, j_end
      aa(j,i) = aa(j-1,i-1) + bb(j,i)
    end do
    !$omp barrier
  end do
  !$omp end parallel
end subroutine tsvc_2_s119_fp64
