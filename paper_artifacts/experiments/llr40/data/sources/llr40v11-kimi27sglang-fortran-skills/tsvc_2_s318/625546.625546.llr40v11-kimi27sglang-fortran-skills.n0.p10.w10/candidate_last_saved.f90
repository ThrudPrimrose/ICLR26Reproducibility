subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, inc, workspace_size
  real(c_double), intent(in) :: a(0:LEN_1D*inc-1)
  real(c_double), intent(out) :: result(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, k, index, t, nt, lo, hi, n
  real(c_double) :: maxv
  real(c_double) :: maxvs(0:127)
  integer(c_int64_t) :: idxs(0:127)

  n = LEN_1D
  nt = omp_get_max_threads()
  if (nt > 128) nt = 128

  maxvs(0) = abs(a(0))
  idxs(0) = 0

  !$omp parallel private(t, lo, hi, i, k) shared(a, maxvs, idxs, n, nt, inc) num_threads(nt)
  t = omp_get_thread_num()
  lo = (n * t) / nt
  hi = (n * (t + 1)) / nt - 1
  if (lo == 0) then
    maxvs(t) = abs(a(0))
    idxs(t) = 0
    k = inc
    do i = 1, hi
      if (abs(a(k)) > maxvs(t)) then
        maxvs(t) = abs(a(k))
        idxs(t) = i
      end if
      k = k + inc
    end do
  else
    k = lo * inc
    maxvs(t) = abs(a(k))
    idxs(t) = lo
    do i = lo, hi
      if (abs(a(k)) > maxvs(t)) then
        maxvs(t) = abs(a(k))
        idxs(t) = i
      end if
      k = k + inc
    end do
  end if
  !$omp end parallel

  maxv = maxvs(0)
  index = idxs(0)
  do i = 1, nt - 1
    if (maxvs(i) > maxv) then
      maxv = maxvs(i)
      index = idxs(i)
    end if
  end do

  result(1) = maxv + dble(index)
end subroutine tsvc_2_s318_fp64
