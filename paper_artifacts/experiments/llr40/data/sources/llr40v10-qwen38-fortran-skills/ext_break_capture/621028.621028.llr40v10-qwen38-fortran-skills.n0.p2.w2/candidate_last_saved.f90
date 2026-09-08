subroutine ext_break_capture_fp64(a, out_index, out_value, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(inout) :: out_index(1)
  real(c_double), intent(inout) :: out_value(1)
  integer(c_int64_t) :: i

  out_index(1) = 0
  out_value(1) = -1.0d0
  do i = 1, len_1d
    if (a(i) > 1.0d0) then
      out_index(1) = i
      out_value(1) = a(i)
      return
    end if
  end do
end subroutine ext_break_capture_fp64
