subroutine wf_diff_skew_fp64(a, n, workspace, workspace_size) bind(C, name="wf_diff_skew_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  real(c_double), intent(inout) :: a(n, n)
  integer(c_int64_t), parameter :: BR = 32
  integer(c_int64_t), parameter :: BC = 128
  integer(c_int64_t) :: s, tr, tc
  integer(c_int64_t) :: tr_first, tr_last, tc_cur
  integer(c_int64_t) :: tr_start, tr_end, tc_start, tc_end
  integer(c_int64_t) :: i, j
  integer(c_int64_t) :: nrows, ncols, ntr, ntc

  nrows = n - 1
  ncols = n - 1
  ntr = (nrows + BR - 1) / BR
  ntc = (ncols + BC - 1) / BC

  !$omp parallel private(s, tr, tc, tr_first, tr_last, tc_cur, tr_start, tr_end, tc_start, tc_end, i, j)
  do s = 1 - ntc, ntr - 1
    tr_first = max(1_c_int64_t, s + 1_c_int64_t)
    tr_last = min(ntr, s + ntc)
    !$omp do schedule(static)
    do tr = tr_first, tr_last
      tc_cur = tr - s
      tr_start = 2 + (tr - 1) * BR
      tr_end = min(n, tr_start + BR - 1)
      tc_start = 1 + (tc_cur - 1) * BC
      tc_end = min(n - 1, tc_start + BC - 1)
      do i = tr_start, tr_end
        !$omp simd
        do j = tc_start, tc_end
          a(j, i) = a(j, i) + a(j, i - 1) + a(j + 1, i - 1)
        end do
      end do
    end do
  end do
  !$omp end parallel
end subroutine wf_diff_skew_fp64
