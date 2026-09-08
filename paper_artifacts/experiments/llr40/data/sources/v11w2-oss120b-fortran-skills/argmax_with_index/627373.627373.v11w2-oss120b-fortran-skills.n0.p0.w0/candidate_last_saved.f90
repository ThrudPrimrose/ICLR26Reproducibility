subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)

  real(c_double) :: max_val
  integer(c_int64_t) :: i, idx

  ! Initialize max_val with first element
  if (LEN_1D > 0) then
    max_val = a(1)
  else
    max_val = 0.0_c_double
  end if

  ! Parallel reduction to find maximum value
  !$omp parallel do reduction(max:max_val) schedule(static) default(none) shared(a, LEN_1D) private(i)
  do i = 2, LEN_1D
    if (a(i) > max_val) then
      max_val = a(i)
    end if
  end do
  !$omp end parallel do

  ! Find the first index of the maximum (1-based)
  idx = LEN_1D   ! sentinel large value
  !$omp parallel do reduction(min:idx) schedule(static) default(none) shared(a, LEN_1D, max_val) private(i)
  do i = 1, LEN_1D
    if (a(i) == max_val) then
      idx = i - 1   ! store zero‑based index for later conversion
    end if
  end do
  !$omp end parallel do

  out_value(1) = max_val
  out_index(1) = idx + 1

end subroutine argmax_with_index_fp64
