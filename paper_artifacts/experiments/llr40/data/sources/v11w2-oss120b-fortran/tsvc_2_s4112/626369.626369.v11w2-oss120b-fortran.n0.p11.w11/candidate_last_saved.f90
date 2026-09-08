subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int) :: i

  !$omp simd
  do i = 1, LEN_1D
    a(i) = a(i) + b(ip(i)) * 2.0_c_double
  end do
end subroutine tsvc_2_s4112_fp64
