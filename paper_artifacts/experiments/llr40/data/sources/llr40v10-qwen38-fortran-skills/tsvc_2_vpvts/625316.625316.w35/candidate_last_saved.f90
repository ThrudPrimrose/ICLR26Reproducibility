subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, s
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)

  real(c_double) :: sc
  integer(kind=8) :: i

  sc = real(s, c_double)

  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = a(i) + b(i) * sc
  end do
end subroutine tsvc_2_vpvts_fp64
