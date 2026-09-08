! TSVC quasi_affine_reduce_odd -- sum(a[i] for i in range(1, LEN_1D, 2))  (C 0-based)
! In Fortran 1-based indexing that is a(2), a(4), ..., a(2*m) with m = LEN_1D/2.
! C ABI: void quasi_affine_reduce_odd_fp64(const double *a, double *out, int64_t LEN_1D, [ws, wsn])
subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(C, name="quasi_affine_reduce_odd_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double),           intent(in)  :: a(len_1d)
  real(c_double),           intent(out) :: out(1)

  real(c_double) :: s0, s1, s2, s3, s4, s5, s6, s7, sr
  integer(c_int64_t) :: m, g, j

  m = len_1d / 2
  g = m / 8
  s0 = 0.0d0; s1 = 0.0d0; s2 = 0.0d0; s3 = 0.0d0
  s4 = 0.0d0; s5 = 0.0d0; s6 = 0.0d0; s7 = 0.0d0; sr = 0.0d0

  !$omp parallel
  !$omp do schedule(static) reduction(+:s0, s1, s2, s3, s4, s5, s6, s7)
  do j = 2, 16*g - 14, 16
    s0 = s0 + a(j)
    s1 = s1 + a(j+2)
    s2 = s2 + a(j+4)
    s3 = s3 + a(j+6)
    s4 = s4 + a(j+8)
    s5 = s5 + a(j+10)
    s6 = s6 + a(j+12)
    s7 = s7 + a(j+14)
  end do
  ! remainder: at most 7 elements (m mod 8), indices 16*g+2 .. 2*m step 2
  !$omp do schedule(static) reduction(+:sr)
  do j = 16*g + 2, 2*m, 2
    sr = sr + a(j)
  end do
  !$omp end parallel

  out(1) = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + sr
end subroutine quasi_affine_reduce_odd_fp64
