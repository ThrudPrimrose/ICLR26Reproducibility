! TSVC s4112: a(i) = a(i) + b(ip(i)) * 2.0   (ip values 1-based for Fortran)
subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d) bind(C, name='tsvc_2_s4112_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout), dimension(*) :: a
  real(c_double), intent(in), dimension(*) :: b
  integer(c_int32_t), intent(in), dimension(*) :: ip
  integer(c_int64_t), value :: len_1d
  integer(c_int64_t) :: i

  do i = 1, len_1d
    a(i) = a(i) + b(ip(i)) * 2.0d0
  end do
end subroutine tsvc_2_s4112_fp64
