subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, S
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t) :: i, n4
  real(c_double) :: rs

  rs = real(S, kind=c_double)
  n4 = iand(LEN_1D, not(3_c_int64_t))
  !$omp parallel do simd schedule(static)
  do i = 1, n4, 4
    a(i)   = a(i)   + b(i)   * rs
    a(i+1) = a(i+1) + b(i+1) * rs
    a(i+2) = a(i+2) + b(i+2) * rs
    a(i+3) = a(i+3) + b(i+3) * rs
  end do
  !$omp end parallel do simd
  do i = n4 + 1, LEN_1D
    a(i) = a(i) + b(i) * rs
  end do
end subroutine tsvc_2_vpvts_fp64
