subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(C, name='tsvc_2_vtvtv_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a, b, c
  integer(c_int64_t), value :: len_1d

  real(c_double), dimension(:), pointer :: av, bv, cv
  integer(c_int64_t) :: i, n

  n = len_1d
  if (n <= 0) return
  call c_f_pointer(a, av, [n])
  call c_f_pointer(b, bv, [n])
  call c_f_pointer(c, cv, [n])

  do i = 1, n
    av(i) = av(i) * bv(i) * cv(i)
  end do
end subroutine tsvc_2_vtvtv_fp64
