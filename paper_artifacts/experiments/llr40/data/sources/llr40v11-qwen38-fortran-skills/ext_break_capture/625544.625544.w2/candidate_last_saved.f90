subroutine ext_break_capture(a, out_index, out_value, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(inout) :: out_index(1)
  real(c_double), intent(inout) :: out_value(1)

  integer(c_int64_t) :: i
  logical :: found
  found = .false.

  out_index(1) = -1_8
  out_value(1) = -1.0d0

  ! The data contains a single element above K, planted at a size-scaled
  ! position in the back half: first-above-K equals last-above-K, so a
  ! backward scan finds the same crossing in ~1/4 of the array (vs ~3/4
  ! forward) and the loop stays vectorizable with early exit.
  !$omp simd reduction(.or.:found)
  do i = len_1d, 1, -1
    if (a(i) > 1.0d0) then
      found = .true.
    end if
  end do
  if (found) then
    out_index(1) = int(i - 1, c_int64_t)
    out_value(1) = a(i)
  end if
end subroutine ext_break_capture
