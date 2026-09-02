! Per-thread dummy counters: give wfds_fence a real, compiler-visible
! memory side effect (so calls cannot be deleted or moved across other
! memory operations) while avoiding false sharing (one cache line each).
module wfds_mod
  implicit none
  integer, public :: wfds_pad(128 * 1024)
end module wfds_mod

! Invisible side-effecting no-op: compiler must keep calls in place.
subroutine wfds_fence(t)
  use wfds_mod
  implicit none
  integer, intent(in) :: t
  wfds_pad(t * 128 + 1) = wfds_pad(t * 128 + 1) + 1
end subroutine wfds_fence

subroutine wf_diff_skew_fp64(a, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: workspace(workspace_size)

  integer :: li, r, c, t, nt, nt_used, total8, base8, rem8, s8, lo, hi, d
  integer, allocatable :: done(:)
  interface
    subroutine wfds_fence(t)
      integer, intent(in) :: t
    end subroutine wfds_fence
  end interface

  li = int(len_2d)
  if (li < 2) return

  ! numpy: a[i,j] = a[i,j] + a[i-1,j] + a[i-1,j+1], i=1..L-1, j=0..L-2
  ! Fortran (axes reversed): a(c,r) = a(c,r) + a(c,r-1) + a(c+1,r-1)
  !   r = 2..L (serial row chain), c = 1..L-1 (unit stride, free).
  ! Each thread owns a column band and marches down the rows.  Thread t's
  ! only cross-thread need is column lo(t+1) of the previous row: the
  ! right neighbour publishes done(t+1) = row as soon as that one column
  ! is stored, so thread t spins briefly before starting the row.
  if (li < 384) then
    do r = 2, li
      do c = 1, li - 1
        a(c, r) = a(c, r) + a(c, r - 1) + a(c + 1, r - 1)
      end do
    end do
  else
    nt = omp_get_max_threads()
    nt_used = nt
    if (nt_used > (li - 1) / 48) nt_used = (li - 1) / 48
    if (nt_used < 1) nt_used = 1
    total8 = (li - 1 + 7) / 8
    base8 = total8 / nt_used
    rem8 = total8 - base8 * nt_used
    allocate(done(nt_used))
    done = 1
    !$omp parallel private(t, lo, hi, r, c, s8, d)
      t = omp_get_thread_num()
      if (t < nt_used) then
        s8 = base8
        if (t < rem8) s8 = s8 + 1
        lo = 1 + 8 * (t * base8 + min(t, rem8))
        hi = lo + s8 * 8 - 1
        if (hi > li - 1) hi = li - 1
        do r = 2, li
          if (t + 1 < nt_used .and. r > 2) then
            do
              d = done(t + 1)
              if (d >= r - 1) exit
              call wfds_fence(t)
            end do
          end if
          a(lo, r) = a(lo, r) + a(lo, r - 1) + a(lo + 1, r - 1)
          call wfds_fence(t)
          done(t) = r
          do c = lo + 1, hi
            a(c, r) = a(c, r) + a(c, r - 1) + a(c + 1, r - 1)
          end do
        end do
      end if
    !$omp end parallel
    deallocate(done)
  end if
end subroutine wf_diff_skew_fp64
