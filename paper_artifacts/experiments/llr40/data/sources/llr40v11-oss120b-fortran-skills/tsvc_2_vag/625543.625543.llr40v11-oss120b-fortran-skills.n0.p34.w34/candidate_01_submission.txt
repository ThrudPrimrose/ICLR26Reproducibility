subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in) :: b(*)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i
  !$omp simd
  do i = 1, LEN_1D
    a(i) = b(ip(i))
  end do
  !$omp end simd
  end subroutine tsvc_2_vag_fp64
