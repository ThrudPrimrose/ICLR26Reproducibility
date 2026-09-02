subroutine wavefront2d_fp64(a, n) bind(C, name='wavefront2d_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
  integer(c_int64_t), value :: n
  real(c_double), intent(inout) :: a(n, n)
  integer(c_int64_t) :: s, i, ilo, ihi

  !$omp parallel default(none) shared(a, n) private(s, i, ilo, ihi)
  do s = 4_c_int64_t, 2_c_int64_t * n
    ilo = max(2_c_int64_t, s - n)
    ihi = min(s - 2_c_int64_t, n)
    !$omp do schedule(static)
    do i = ilo, ihi
      a(s - i, i) = 0.25_c_double * (a(s - i, i) + a(s - i, i - 1_c_int64_t) + &
                                      a(s - i - 1_c_int64_t, i) + a(s - i - 1_c_int64_t, i - 1_c_int64_t))
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wavefront2d_fp64
