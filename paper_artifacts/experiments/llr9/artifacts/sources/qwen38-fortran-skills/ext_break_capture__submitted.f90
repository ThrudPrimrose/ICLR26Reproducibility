subroutine ext_break_capture_fp64(a, out_index, out_value, k, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: k
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(inout) :: out_index(1)
  real(c_double), intent(inout) :: out_value(1)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: first, i, n
  real(c_double) :: kd

  kd = real(k, c_double)
  n = len_1d
  if (n < 16384) then
    ! small: single-thread vectorized masked scan with early exit
    first = n + 1
    do i = 1, n
      if (a(i) > kd) then
        first = i
        exit
      end if
    end do
  else
    ! large: minimum matching index over the whole array, threaded + SIMD
    first = n + 1
    !$omp parallel do reduction(min:first)
    do i = 1, n
      first = min(first, merge(i, n + 1, a(i) > kd))
    end do
  end if
  if (first <= n) then
    out_index(1) = first - 1
    out_value(1) = a(first)
  else
    out_index(1) = -1
    out_value(1) = -1.0d0
  end if
end subroutine ext_break_capture_fp64
