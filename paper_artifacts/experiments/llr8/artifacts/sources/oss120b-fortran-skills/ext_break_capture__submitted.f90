module ext_break_capture_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, k, len_1d, workspace, workspace_size) bind(C, name="ext_break_capture_fp64")
    ! Arguments
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(inout) :: out_index(1)
    real(c_double), intent(inout) :: out_value(1)
    integer(c_int64_t), value, intent(in) :: k
    integer(c_int64_t), value, intent(in) :: len_1d
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    ! Local variables
    integer(c_int64_t) :: i
    ! Initialize output to sentinel values
    out_index(1) = -1_c_int64_t
    out_value(1) = -1.0_c_double
    do i = 1, len_1d
      if (a(i) > real(k, kind=c_double)) then
        out_index(1) = i - 1_c_int64_t
        out_value(1) = a(i)
        exit
      end if
    end do
  end subroutine ext_break_capture_fp64
end module ext_break_capture_mod
