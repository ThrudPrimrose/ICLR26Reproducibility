subroutine tsvc_2_s318_fp64(a, result, len_1d, inc) bind(c, name="tsvc_2_s318_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in)    :: a(*)
  real(c_double), intent(out)   :: result(1)
  integer(c_int64_t), value, intent(in) :: len_1d, inc

  integer(c_int64_t) :: i, index
  real(c_double) :: maxv, v

  index = 0
  maxv = abs(a(1))
  do i = 1, len_1d - 1
     v = abs(a(i * inc + 1))
     if (v > maxv) then
        index = i
        maxv = v
     end if
  end do
  result(1) = maxv + real(index, c_double)
end subroutine tsvc_2_s318_fp64
