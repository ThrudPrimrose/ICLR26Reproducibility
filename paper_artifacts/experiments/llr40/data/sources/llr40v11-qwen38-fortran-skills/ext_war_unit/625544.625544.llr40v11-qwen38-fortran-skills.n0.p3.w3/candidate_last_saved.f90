subroutine ext_war_unit_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(inout) :: workspace(workspace_size)

  integer, parameter :: TIL = 16384
  integer(c_int64_t) :: m, nt, t, t0, t1, kk, i
  real(c_double), allocatable :: e(:)

  m = len_1d - 1
  if (m > 0) then
    nt = (m + TIL - 1) / TIL
    allocate(e(nt))
    !$omp parallel do shared(e) private(t0, t1, kk) schedule(static)
    do t = 1, nt
      t0 = (t - 1) * TIL + 1
      t1 = min(t0 + TIL - 1, m)
      kk = t1 - t0 + 1
      e(t) = a(t0 + kk)
    end do
    !$omp parallel do shared(e, a, b, m) private(t0, t1, kk, i) schedule(static)
    do t = 1, nt
      t0 = (t - 1) * TIL + 1
      t1 = min(t0 + TIL - 1, m)
      kk = t1 - t0 + 1
      if (kk >= 2) then
        !$omp simd
        do i = t0, t1 - 1
          a(i) = a(i + 1) + b(i)
        end do
      end if
      a(t1) = e(t) + b(t1)
    end do
    deallocate(e)
  end if
end subroutine ext_war_unit_fp64
