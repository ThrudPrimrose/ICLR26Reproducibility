module ext_break_capture_inout_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, len_1d) bind(C, name="ext_break_capture_fp64")
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(inout) :: out_index
    real(c_double), intent(inout) :: out_value
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    out_index = -1_c_int64_t
    out_value = -1.0_c_double
    do i = 1, len_1d
      if (a(i) > 1.0_c_double) then
        out_index = i - 1_c_int64_t
        out_value = a(i)
        exit
      end if
    end do
  end subroutine ext_break_capture_fp64
end module ext_break_capture_inout_mod
