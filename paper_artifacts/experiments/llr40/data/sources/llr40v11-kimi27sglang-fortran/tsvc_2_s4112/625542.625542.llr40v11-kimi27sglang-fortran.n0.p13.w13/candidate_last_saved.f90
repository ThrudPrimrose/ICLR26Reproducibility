subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(c, name='tsvc_2_s4112_fp64')
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do simd schedule(static, 256)
  do i = 1, LEN_1D
    a(i) = a(i) + b(ip(i)) * 2.0_c_double
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s4112_fp64
