subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(C, name="tsvc_2_s235_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)

  integer(c_int64_t) :: ic, ir, n, lo, hi, t, nt
  n = len_2d

  !$omp parallel do simd schedule(static)
  do ic = 1, n
    a(ic) = a(ic) + b(ic) * c(ic)
  end do

  !$omp parallel private(t, nt, lo, hi, ic, ir)
  t = omp_get_thread_num()
  nt = omp_get_num_threads()
  lo = (n * t) / nt + 1
  hi = (n * (t + 1)) / nt
  do ir = 2, n
    do ic = lo, hi
      aa(ic, ir) = aa(ic, ir-1) + bb(ic, ir) * a(ic)
    end do
  end do
  !$omp end parallel
end subroutine tsvc_2_s235_fp64
