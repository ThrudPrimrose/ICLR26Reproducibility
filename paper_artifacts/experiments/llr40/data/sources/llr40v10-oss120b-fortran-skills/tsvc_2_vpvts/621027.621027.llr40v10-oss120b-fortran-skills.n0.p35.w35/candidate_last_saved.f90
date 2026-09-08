subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, S
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  
    real(c_double) :: factor
  integer(c_int64_t) :: i
    factor = real(S, c_double)
  do concurrent (i = 1:LEN_1D)
    a(i) = a(i) + b(i) * factor
  end do

end subroutine tsvc_2_vpvts_fp64
