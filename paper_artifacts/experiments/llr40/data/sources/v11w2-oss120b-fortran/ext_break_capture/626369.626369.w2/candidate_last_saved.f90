module ext_break_capture_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(*)
    real(c_double), intent(out) :: out_value(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double), parameter :: k = 1.0_c_double
    out_index(1) = -1_c_int64_t
    out_value(1) = -1.0_c_double
    integer(c_int64_t) :: min_idx
    min_idx = LEN_1D
    !$omp parallel do reduction(min:min_idx) schedule(static)
    do i = 0, LEN_1D - 1
      if (a(i+1) > k) then
        min_idx = i
      end if
    end do
    !$omp end parallel do
    if (min_idx < LEN_1D) then
      out_index(1) = min_idx + 1_c_int64_t
      out_value(1) = a(min_idx+1)
    else
      out_index(1) = -1_c_int64_t
      out_value(1) = -1.0_c_double
    end if
  end subroutine ext_break_capture_fp64
end module ext_break_capture_mod
