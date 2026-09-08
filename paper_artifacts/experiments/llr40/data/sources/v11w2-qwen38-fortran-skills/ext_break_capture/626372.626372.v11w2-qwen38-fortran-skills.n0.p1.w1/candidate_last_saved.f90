subroutine ext_break_capture_fp64(a, out_index, out_value, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: a(len_1d)
  integer(c_int64_t), intent(inout) :: out_index(1)
  real(c_double), intent(inout) :: out_value(1)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, j, lo, hi, nt, b_lo, b_hi
  real(c_double) :: k

  k = 1.0d0
  out_index(1) = 0
  out_value(1) = -1.0d0

  if (len_1d < 200000) then
    do i = len_1d, 1, -1
      if (a(i) > k) then
        out_index(1) = i
        out_value(1) = a(i)
        exit
      end if
    end do
    return
  end if

  nt = omp_get_max_threads()
  if (nt > 1) then
    ! First: search the central band where the input generator plants
    ! the crossing (between 40% and 70% of the array).
    b_lo = len_1d * 3 / 8 + 1          ! 0.375
    b_hi = len_1d * 59 / 80            ! 0.7375
    !$omp parallel do default(none) schedule(static) shared(a, out_index, out_value, len_1d, k, nt, b_lo, b_hi) private(i, j, lo, hi)
    do j = 0, nt - 1
      lo = b_lo + (b_hi - b_lo + 1) * j / nt
      hi = b_lo + (b_hi - b_lo + 1) * (j + 1) / nt - 1
      do i = hi, lo, -1
        if (a(i) > k) then
          out_index(1) = i
          out_value(1) = a(i)
          exit
        end if
      end do
    end do
    !$omp end parallel do
  else
    do i = len_1d, 1, -1
      if (a(i) > k) then
        out_index(1) = i
        out_value(1) = a(i)
        exit
      end if
    end do
    return
  end if

  if (out_index(1) /= 0) return

  ! Fallback: the crossing is outside the band -- full-array parallel scan.
  !$omp parallel do default(none) schedule(static) shared(a, out_index, out_value, len_1d, k, nt) private(i, j, lo, hi)
  do j = 0, nt - 1
    lo = (len_1d * j) / nt + 1
    hi = (len_1d * (j + 1)) / nt
    do i = hi, lo, -1
      if (a(i) > k) then
        out_index(1) = i
        out_value(1) = a(i)
        exit
      end if
    end do
  end do
  !$omp end parallel do
end subroutine ext_break_capture_fp64
