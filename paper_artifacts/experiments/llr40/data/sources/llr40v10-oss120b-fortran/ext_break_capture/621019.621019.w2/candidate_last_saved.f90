module ext_break_capture_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
    real(C_DOUBLE), intent(in) :: a(*)
    integer(C_INT64_T), intent(out) :: out_index(*)
    real(C_DOUBLE), intent(out) :: out_value(*)
    integer(C_INT64_T), value :: LEN_1D
    integer(C_INT64_T) :: i, len_minus_one
    real(C_DOUBLE), parameter :: K = 1.0_C_DOUBLE
    out_index(1) = -1_C_INT64_T
    out_value(1) = -1.0_C_DOUBLE
    len_minus_one = LEN_1D - 1_C_INT64_T
    do i = 0, len_minus_one
        if (a(i+1) > K) then
            out_index(1) = i
            out_value(1) = a(i+1)
            exit
        end if
    end do
  end subroutine ext_break_capture_fp64
end module ext_break_capture_mod
