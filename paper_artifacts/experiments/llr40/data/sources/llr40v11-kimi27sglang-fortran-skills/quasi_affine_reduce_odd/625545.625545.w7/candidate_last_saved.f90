subroutine quasi_affine_reduce_odd(a, out, LEN_1D) bind(C, name="quasi_affine_reduce_odd_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(1)
  print *, 'LEN_1D=', LEN_1D
  flush(6)
  out(1) = 0.0_c_double
end subroutine quasi_affine_reduce_odd
