subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: sum_out
  real(c_double) :: part(8, 256)
  real(c_double) :: s, ps
  integer(c_int64_t) :: i, lo, hi
  integer :: t, nt, maxt
  integer(c_int64_t), parameter :: grain = 8192_c_int64_t

  s = 0.0_c_double
  if (LEN_1D < grain) then
    !$omp simd reduction(+:s)
    do i = 1, LEN_1D
      s = s + a(i)
    end do
  else
    nt = omp_get_max_threads()
    maxt = int(LEN_1D / grain)
    if (maxt < 1) maxt = 1
    if (nt > maxt) nt = maxt
    if (nt > 256) nt = 256
    !$omp parallel do schedule(static) private(i, lo, hi, ps) shared(part) num_threads(nt)
    do t = 1, nt
      lo = (LEN_1D * (t - 1_c_int64_t)) / nt + 1_c_int64_t
      hi = (LEN_1D * t) / nt
      ps = 0.0_c_double
      !$omp simd reduction(+:ps)
      do i = lo, hi
        ps = ps + a(i)
      end do
      part(1, t) = ps
    end do
    s = sum(part(1, 1:nt))
  end if
  sum_out = s
end subroutine tsvc_2_s311_fp64
