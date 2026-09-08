subroutine argmax_fp64_impl(a, out_index, out_value, LEN_1D) bind(C, name='argmax_with_index_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), value :: LEN_1D
  real(c_double) :: x
  integer(c_int64_t) :: i, idx
  if (LEN_1D <= 0) then
     out_value = 0.0d0
     out_index = 0
     return
  end if
  x = a(1)
  idx = 1
  do i = 2, LEN_1D
     if (a(i) > x) then
        x = a(i)
        idx = i
     end if
  end do
  out_value = x
  out_index = idx
end subroutine argmax_fp64_impl
