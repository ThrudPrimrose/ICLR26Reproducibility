subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D

  integer(c_int64_t) :: i
  integer(c_int64_t) :: first_idx
  real(c_double), parameter :: K = 1.0_c_double
  integer(c_int64_t), parameter :: PARALLEL_THRESHOLD = 8192_c_int64_t

  first_idx = LEN_1D + 1

  if (LEN_1D < PARALLEL_THRESHOLD) then
    do i = 1, LEN_1D
      if (a(i) > K) then
        first_idx = i
        exit
      end if
    end do
  else
    !$omp parallel do reduction(min:first_idx) schedule(static)
    do i = 1, LEN_1D
      if (a(i) > K) then
        first_idx = i
      end if
    end do
    !$omp end parallel do
  end if

  if (first_idx <= LEN_1D) then
    out_index(1) = first_idx
    out_value(1) = a(first_idx)
  else
    out_index(1) = 0_c_int64_t
    out_value(1) = -1.0_c_double
  end if
end subroutine ext_break_capture_fp64
