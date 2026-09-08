subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)
  integer(c_int64_t) :: i
  real(c_double) :: max_val
  integer(c_int64_t) :: idx

  max_val = a(1)
  idx = 0_c_int64_t
  do i = 2, LEN_1D
    if (a(i) > max_val) then
      max_val = a(i)
      idx = i - 1_c_int64_t
    end if
  end do

  out_value(1) = max_val
  out_index(1) = idx
end subroutine argmax_with_index_fp64
