subroutine ext_break_capture_fp64(a_p, oi_p, ov_p, n) bind(C, name='ext_break_capture_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a_p, oi_p, ov_p
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), dimension(:), pointer :: a
  integer(c_int64_t), dimension(:), pointer :: out_index
  real(c_double), dimension(:), pointer :: out_value
  integer(c_int64_t) :: i
  real(c_double) :: k
  k = 1.0d0
  call c_f_pointer(a_p, a, [n])
  call c_f_pointer(oi_p, out_index, [1])
  call c_f_pointer(ov_p, out_value, [1])
  out_index(1) = int(-1, c_int64_t)
  out_value(1) = -1.0d0
  do i = int(1, c_int64_t), n
    if (a(i) > k) then
      out_index(1) = i
      out_value(1) = a(i)
      exit
    end if
  end do
end subroutine ext_break_capture_fp64
