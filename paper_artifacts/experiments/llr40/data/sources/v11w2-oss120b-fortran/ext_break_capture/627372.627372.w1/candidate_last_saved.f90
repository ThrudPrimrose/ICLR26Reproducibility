module ext_break_capture_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
    real(c_double), intent(in) :: a(0:*)
    integer(c_int64_t), intent(out) :: out_index
    real(c_double), intent(out) :: out_value
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    out_index = -1_c_int64_t
    out_value = -1.0_c_double
    if (LEN_1D <= 0) return
    do i = 0_c_int64_t, LEN_1D - 1
      if (a(i) > 1.0_c_double) then
        out_index = i
        out_value = a(i)
        exit
      end if
    end do
  end subroutine ext_break_capture_fp64
end module ext_break_capture_mod
