subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in)    :: cc(len_2d, len_2d)
  integer(c_int64_t) :: n, lo, hi, t

  n = len_2d
  if (n < 9) return

  !$omp parallel
  ! per-thread contiguous first-index block; elementwise over block per chain step
  lo = 9 + (n - 8) * omp_get_thread_num() / omp_get_num_threads()
  hi = 9 + (n - 8) * (omp_get_thread_num() + 1) / omp_get_num_threads() - 1
  do t = 9, n
    aa(lo:hi, t) = aa(lo:hi, t - 1) + cc(lo:hi, t)
    bb(lo:hi, t) = bb(lo:hi, t - 1) + cc(lo:hi, t)
  end do
  !$omp end parallel
end subroutine tsvc_2_s2233_fp64
