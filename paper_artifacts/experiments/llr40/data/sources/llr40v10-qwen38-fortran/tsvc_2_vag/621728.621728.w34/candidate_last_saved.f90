subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value, intent(in) :: b
  type(c_ptr), value, intent(in) :: ip
  integer(c_int64_t), value, intent(in) :: len_1d

  real(c_double), pointer :: pa(:)
  real(c_double), pointer :: pb(:)
  integer(c_int32_t), pointer :: ppi(:)
  integer(c_int64_t) :: i

  call c_f_pointer(a, pa, [len_1d])
  call c_f_pointer(b, pb, [len_1d])
  call c_f_pointer(ip, ppi, [len_1d])

  do i = 1, len_1d
    pa(i) = pb(ppi(i))
  end do
end subroutine tsvc_2_vag_fp64
