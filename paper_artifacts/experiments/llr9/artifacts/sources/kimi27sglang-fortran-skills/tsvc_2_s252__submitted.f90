subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: nt, tid, lo, hi, chunk, i
  real(c_double) :: s, prev_s

  if (LEN_1D <= 0) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  chunk = (LEN_1D + nt - 1) / nt

  !$omp parallel do private(tid, lo, hi, s, prev_s) schedule(static)
  do tid = 0, nt - 1
    lo = tid * chunk + 1
    hi = min((tid + 1) * chunk, LEN_1D)
    if (lo > hi) cycle
    if (lo == 1) then
      prev_s = 0.0d0
    else
      prev_s = b(lo - 1) * c(lo - 1)
    end if
    do i = lo, hi
      s = b(i) * c(i)
      a(i) = s + prev_s
      prev_s = s
    end do
  end do
  !$omp end parallel do
end subroutine tsvc_2_s252_fp64
