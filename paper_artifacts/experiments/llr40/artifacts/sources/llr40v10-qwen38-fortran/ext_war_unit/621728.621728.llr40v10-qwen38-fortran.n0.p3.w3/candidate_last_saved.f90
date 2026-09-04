subroutine ext_war_unit_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i, lo, hi, n, nt, t, per
  real(c_double) :: bv

  n = len_1d
  if (n <= 1) return

  ! Small sizes: the fork/join of a thread team would cost more than the loop.
  if (n < 262144) then
     do i = 1, n-1
        a(i) = a(i+1) + b(i)
     end do
     return
  end if

  ! a(i) = a(i+1) + b(i) is elementwise once the ORIGINAL a(i+1) is read: the only
  ! carried "dependence" is a false WAR.  Per thread: save the right boundary
  ! (a(hi+1), still original since nothing has written yet), barrier, then run the
  ! vectorizable ascending loop, using the saved value for the block's last element.
  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  per = (n + nt - 1) / nt

  !$omp parallel num_threads(nt) private(t, lo, hi, bv, i)
  t = omp_get_thread_num()
  lo = t * per + 1
  hi = min(lo + per - 1, n)
  if (lo <= hi) then
     if (hi < n) bv = a(hi + 1)
     !$omp barrier
     do i = lo, hi - 1
        a(i) = a(i+1) + b(i)
     end do
     if (hi < n) a(hi) = bv + b(hi)
  end if
  !$omp end parallel
end subroutine
