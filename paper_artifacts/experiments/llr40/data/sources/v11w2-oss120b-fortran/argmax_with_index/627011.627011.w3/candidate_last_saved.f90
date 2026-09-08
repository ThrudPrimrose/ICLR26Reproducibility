subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i
  real(c_double) :: x
  integer(c_int64_t) :: idx
  x = a(1)
  idx = 0
  do i = 2, LEN_1D
    if (a(i) > x) then
      x = a(i)
      idx = i-1
    end if
  end do
  out_value(1) = x
  out_index(1) = idx
end subroutine argmax_with_index_fp64
