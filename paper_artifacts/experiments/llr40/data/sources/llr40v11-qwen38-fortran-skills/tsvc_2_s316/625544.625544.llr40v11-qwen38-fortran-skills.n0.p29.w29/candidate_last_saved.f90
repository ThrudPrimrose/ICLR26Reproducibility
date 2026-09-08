subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name="tsvc_2_s316_fp64")
  use iso_c_binding
  use omp_lib
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)
  real(c_double) :: part(4096), local
  integer(c_int64_t) :: t, lo, hi, i, nt

  if (len_1d <= 1) then
    result(1) = a(1)
    return
  end if
  nt = omp_get_max_threads()
  !$omp parallel private(t, lo, hi, i, local)
  t = omp_get_thread_num()
  lo = (len_1d * t) / nt + 1
  hi = (len_1d * (t + 1)) / nt
  local = a(lo)
  do i = lo + 1, hi
    local = min(local, a(i))
  end do
  part(t + 1) = local
  !$omp end parallel
  result(1) = minval(part(1:nt))
end subroutine tsvc_2_s316_fp64
