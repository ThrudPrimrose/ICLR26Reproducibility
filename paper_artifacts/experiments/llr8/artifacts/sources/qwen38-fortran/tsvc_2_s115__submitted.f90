subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name='tsvc_2_s115_fp64')
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: aa(*)
  integer(c_int64_t) :: N, j, k, t0, t1, a0, a1
  real(c_double) :: aj
  integer :: tid, nt
  N = LEN_2D
  !$omp parallel num_threads(16) private(tid,nt,t0,t1,a0,a1,aj,j,k)
    tid = omp_get_thread_num()
    nt = omp_get_num_threads()
    t0 = int(tid * N / nt)
    t1 = int((tid+1) * N / nt)
    do j = 0, N-1
      aj = a(j+1)
      a0 = max(j+1, t0)
      a1 = min(N, t1)
      do k = a0, a1-1
        a(k+1) = a(k+1) - aa(j*N + k + 1) * aj
      end do
      !$omp barrier
    end do
  !$omp end parallel
end subroutine
