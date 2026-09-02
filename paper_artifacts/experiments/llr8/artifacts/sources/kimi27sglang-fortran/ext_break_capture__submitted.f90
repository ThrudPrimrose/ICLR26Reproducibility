subroutine ext_break_capture_fp64(a, out_index, out_value, n) bind(c, name='ext_break_capture_fp64')
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(in) :: a(1:*)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), intent(in), value :: n
  real(c_double), parameter :: k = 1.0_c_double
  integer(c_int64_t) :: i

  out_index = -1_c_int64_t
  out_value = -1.0_c_double
  do i = n, 1, -1
    if (a(i) > k) then
      out_index = i
      out_value = a(i)
      exit
    end if
  end do
end subroutine ext_break_capture_fp64
