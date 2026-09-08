subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D, K) bind(C)
  use iso_c_binding, only: c_double, c_int64_t
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: K
  real(c_double), intent(in) :: a(0:LEN_1D-1)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)

  integer(c_int64_t) :: i
  real(c_double) :: Kd

  Kd = real(K, c_double)

  ! Initialize sentinels
  out_index(1) = -1_c_int64_t
  out_value(1) = -1.0_c_double

  do i = 0, LEN_1D-1
    if (a(i) > Kd) then
      out_index(1) = i
      out_value(1) = a(i)
      exit
    end if
  end do

end subroutine ext_break_capture_fp64
