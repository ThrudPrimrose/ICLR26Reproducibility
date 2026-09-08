subroutine tsvc_2_s231_fp64(aa, bb, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  print *, 'len_2d =', len_2d
  flush(6)
end subroutine tsvc_2_s231_fp64
