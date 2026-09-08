subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i
  do i = 1, LEN_1D
    a(i) = b(ip(i))
  end do
end subroutine tsvc_2_vag_fp64
