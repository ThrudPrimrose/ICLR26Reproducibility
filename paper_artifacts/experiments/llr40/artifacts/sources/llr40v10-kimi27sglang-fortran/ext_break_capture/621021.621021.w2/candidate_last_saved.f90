subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D, workspace, workspace_size) bind(c, name="ext_break_capture_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(inout) :: out_index(*)
  real(c_double), intent(inout) :: out_value(*)
  integer(c_int64_t), intent(in), value :: LEN_1D
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), intent(in), value :: workspace_size
  integer(c_int64_t) :: i
  out_index(1) = -1_c_int64_t
  out_value(1) = -1.0_c_double
  do i = 1, LEN_1D
     if (a(i) > 1.0_c_double) then
        out_index(1) = i - 1_c_int64_t
        out_value(1) = a(i)
        return
     end if
  end do
end subroutine ext_break_capture_fp64
