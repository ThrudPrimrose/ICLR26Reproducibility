subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name='argmax_with_index_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int64_t) :: idx, i, lo, hi, t, nt, chunk
  real(c_double) :: x, lmax
  real(c_double) :: tmax(0:127)
  integer(c_int64_t) :: tidx(0:127)

  idx = 1
  x = a(1)

  if (LEN_1D > 4096) then
     !$omp parallel private(t, lo, hi, chunk, lmax, i)
     nt = omp_get_num_threads()
     t = omp_get_thread_num()
     chunk = (LEN_1D + int(nt, c_int64_t) - 1) / int(nt, c_int64_t)
     lo = 1 + t * chunk
     hi = min(lo + chunk - 1, LEN_1D)
     if (lo <= hi) then
        i = maxloc(a(lo:hi), dim=1)
        lmax = a(lo + i - 1)
        tmax(t) = lmax
        tidx(t) = lo + i - 1
     else
        tmax(t) = x
        tidx(t) = idx
     end if
     !$omp end parallel

     ! global reduction
     do t = 0, nt - 1
        if (tmax(t) > x) then
           x = tmax(t)
           idx = tidx(t)
        end if
     end do
  else
     i = maxloc(a(1:LEN_1D), dim=1)
     x = a(i)
     idx = i
  end if

  out_value(1) = x
  out_index(1) = idx
end subroutine argmax_with_index_fp64
