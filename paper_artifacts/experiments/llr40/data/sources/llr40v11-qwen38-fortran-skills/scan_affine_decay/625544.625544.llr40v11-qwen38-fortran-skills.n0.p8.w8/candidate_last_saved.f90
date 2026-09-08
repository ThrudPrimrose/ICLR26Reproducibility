subroutine scan_affine_decay_fp64(c, x, y, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: c(len_1d), x(len_1d)
  real(c_double), intent(inout) :: y(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  real(c_double) :: y0, pa, pb, ta, tb
  real(c_double), allocatable :: atot(:), btot(:), vstart(:)
  real(c_double), pointer :: wA(:)
  logical :: use_ws
  integer :: nt, b
  integer(c_int64_t) :: n, nscan, i, lo, hi, chunk

  n = len_1d
  if (n < 2) return

  y0 = y(1)
  nscan = n - 1
  nt = omp_get_max_threads()

  if (nscan < 4096 * nt) then
    ! serial path for tiny problems
    do i = 2, n
      y(i) = c(i) * y(i - 1) + x(i)
    end do
    return
  end if

  use_ws = c_associated(workspace) .and. workspace_size >= 8 * n
  if (use_ws) call c_f_pointer(workspace, wA, [n])

  allocate(atot(nt), btot(nt), vstart(nt))
  chunk = (nscan + nt - 1) / nt
  chunk = (chunk + 7) / 8 * 8

  if (use_ws) then
    ! Phase 1: per-chunk local scan (chained); store local value in y,
    ! local product prefix in workspace array wA.
    !$omp parallel do
    do b = 1, nt
      lo = 1 + (b - 1) * chunk + 1
      hi = min(lo + chunk - 1, n)
      ta = 1.0d0
      tb = 0.0d0
      do i = lo, hi
        tb = c(i) * tb + x(i)
        ta = ta * c(i)
        y(i) = tb
        wA(i) = ta
      end do
      atot(b) = ta
      btot(b) = tb
    end do

    ! Phase 2: serial scan over chunk totals
    pa = 1.0d0
    pb = 0.0d0
    vstart(1) = y0
    do b = 1, nt - 1
      pb = atot(b) * pb + btot(b)
      pa = atot(b) * pa
      vstart(b + 1) = pa * y0 + pb
    end do

    ! Phase 3: elementwise apply of chunk-start value (vectorizable)
    !$omp parallel do
    do b = 1, nt
      lo = 1 + (b - 1) * chunk + 1
      hi = min(lo + chunk - 1, n)
      do i = lo, hi
        y(i) = wA(i) * vstart(b) + y(i)
      end do
    end do
  else
    ! Phase 1 (no scratch): local values in y only
    !$omp parallel do
    do b = 1, nt
      lo = 1 + (b - 1) * chunk + 1
      hi = min(lo + chunk - 1, n)
      ta = 1.0d0
      tb = 0.0d0
      do i = lo, hi
        tb = c(i) * tb + x(i)
        ta = ta * c(i)
        y(i) = tb
      end do
      atot(b) = ta
      btot(b) = tb
    end do

    ! Phase 2: serial scan over chunk totals
    pa = 1.0d0
    pb = 0.0d0
    vstart(1) = y0
    do b = 1, nt - 1
      pb = atot(b) * pb + btot(b)
      pa = atot(b) * pa
      vstart(b + 1) = pa * y0 + pb
    end do

    ! Phase 3: re-derive local product prefix on the fly
    !$omp parallel do
    do b = 1, nt
      lo = 1 + (b - 1) * chunk + 1
      hi = min(lo + chunk - 1, n)
      ta = 1.0d0
      do i = lo, hi
        ta = ta * c(i)
        y(i) = ta * vstart(b) + y(i)
      end do
    end do
  end if

  deallocate(atot, btot, vstart)
end subroutine scan_affine_decay_fp64
