! Variant: 48 threads pinned to NUMA node 0 CPUs (0-23, 96-119)
subroutine tsvc_2_s316_fp64(a, r, len_1d) bind(C, name='tsvc_2_s316_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_int, c_size_t, c_ptr, c_loc
  use, intrinsic :: omp_lib, only: omp_get_num_threads, omp_get_thread_num
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), dimension(len_1d), intent(in) :: a
  real(c_double), dimension(1), intent(out) :: r
  real(c_double) :: best, x1, x2, x3, x4
  integer(c_int64_t) :: n, chunk, s, e, i, n4
  integer :: nt, tid, cpulist(48), k
  integer(c_int64_t), target :: cm(4)
  integer(c_int) :: rc
  interface
    integer(c_int) function csetaff_(pid, size, mask) bind(C, name='sched_setaffinity')
      use iso_c_binding
      integer(c_int), value, intent(in) :: pid
      integer(c_size_t), value, intent(in) :: size
      type(c_ptr), value, intent(in) :: mask
    end function
  end interface

  do k = 1, 48
     cpulist(k) = merge(96 + k - 25, k - 1, k > 24)
  end do

  n = len_1d
  if (n == 1) then
     r(1) = a(1)
     return
  end if

  best = a(1)
!$omp parallel num_threads(48) &
!$omp      default(none) shared(a, n, cpulist) &
!$omp      private(nt, tid, chunk, s, e, i, n4, x1, x2, x3, x4, cm, rc) &
!$omp      reduction(min:best)
  nt  = omp_get_num_threads()
  tid = omp_get_thread_num()
  cm = 0
  cm(cpulist(tid+1)/64 + 1) = 2**(mod(cpulist(tid+1), 8))
  rc = csetaff_(0, int(32, c_size_t), c_loc(cm))
  chunk = (n + nt - 1) / nt
  s = min(tid * chunk + 1, n)
  e = min(s + chunk - 1, n)

  x1 = a(1); x2 = a(1); x3 = a(1); x4 = a(1)
  n4 = (e - s + 1) / 4
  do i = s, s + 4*(n4-1), 4
     x1 = min(x1, a(i))
     x2 = min(x2, a(i+1))
     x3 = min(x3, a(i+2))
     x4 = min(x4, a(i+3))
  end do
  i = s + 4*n4
  do while (i <= e)
     best = min(best, a(i))
     i = i + 1
  end do
  best = min(best, min(x1, x2))
  best = min(best, min(x3, x4))
!$omp end parallel

  r(1) = best
end subroutine
