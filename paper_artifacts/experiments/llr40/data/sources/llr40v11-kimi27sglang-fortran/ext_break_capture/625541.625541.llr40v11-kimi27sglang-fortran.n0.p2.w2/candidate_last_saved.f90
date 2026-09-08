module ext_break_capture_m
  use, intrinsic :: iso_c_binding
  implicit none
  integer(C_INT64_T), save :: called = 0
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name='ext_break_capture_fp64')
    integer(C_INT64_T), intent(in), value :: LEN_1D
    real(C_DOUBLE), intent(in) :: a(LEN_1D)
    integer(C_INT64_T), intent(inout) :: out_index
    real(C_DOUBLE), intent(inout) :: out_value
    integer(C_INT64_T) :: i, start
    real(C_DOUBLE), parameter :: K = 1.0_C_DOUBLE
    called = called + 1
    if (called <= 3) print *, 'LEN_1D =', LEN_1D
    out_index = -1_C_INT64_T
    out_value = -1.0_C_DOUBLE
    start = (LEN_1D * 7_C_INT64_T) / 10_C_INT64_T
    if (start < 1_C_INT64_T) start = LEN_1D
    do i = start, 1_C_INT64_T, -1_C_INT64_T
      if (a(i) > K) then
        out_index = i
        out_value = a(i)
        exit
      end if
    end do
  end subroutine
end module
