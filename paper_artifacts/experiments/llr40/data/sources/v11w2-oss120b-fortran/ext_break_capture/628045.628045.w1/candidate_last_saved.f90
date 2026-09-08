subroutine ext_break_capture_fp64(a_ptr, out_index_ptr, out_value_ptr, LEN_1D) bind(C, name='ext_break_capture_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a_ptr
  type(c_ptr), value :: out_index_ptr
  type(c_ptr), value :: out_value_ptr
  integer(c_int64_t), value :: LEN_1D
  real(c_double), pointer :: a(:)
  integer(c_int64_t), pointer :: out_index
  real(c_double), pointer :: out_value
  integer(c_int64_t) :: i
  integer(c_size_t) :: len
  real(c_double) :: k

  len = LEN_1D
  call c_f_pointer(a_ptr, a, [len])
  call c_f_pointer(out_index_ptr, out_index)
  call c_f_pointer(out_value_ptr, out_value)

  out_index = -1_c_int64_t
  out_value = -1.0_c_double
  k = 1.0d0

  do i = 0, LEN_1D - 1
    if (a(i+1) > k) then
      out_index = i
      out_value = a(i+1)
      exit
    end if
  end do
end subroutine ext_break_capture_fp64
