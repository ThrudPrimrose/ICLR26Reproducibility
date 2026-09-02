subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(C, name="tsvc_2_s255_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double) :: x, y
  integer(c_int64_t) :: i

  if (len_1d <= 0) return
  if (len_1d == 1) then
     a(1) = (b(1) + b(1) + b(1)) * 0.333d0
     return
  end if
  x = b(len_1d)
  y = b(len_1d - 1)
  do i = 1, len_1d
     a(i) = (b(i) + x + y) * 0.333d0
     y = x
     x = b(i)
  end do
end subroutine tsvc_2_s255_fp64
