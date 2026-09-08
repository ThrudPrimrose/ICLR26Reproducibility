subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, LEN_2D) bind(C, name='tsvc_2_s2275_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D), aa(LEN_2D, LEN_2D)
  real(c_double), intent(in)    :: b(LEN_2D), bb(LEN_2D, LEN_2D)
  real(c_double), intent(in)    :: c(LEN_2D), cc(LEN_2D, LEN_2D)
  real(c_double), intent(in)    :: d(LEN_2D)
  print *, 'LEN_2D =', LEN_2D
  a(1) = a(1)
end subroutine tsvc_2_s2275_fp64
