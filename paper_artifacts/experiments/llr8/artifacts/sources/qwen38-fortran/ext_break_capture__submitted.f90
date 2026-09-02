subroutine ext_break_capture_fp64(a, out_index, out_value, k, len_1d, ws, ws_size) &
      bind(C, name='ext_break_capture_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index(*)
  real(c_double), intent(out) :: out_value(*)
  integer(c_int64_t), value :: k
  integer(c_int64_t), value :: len_1d
  type(c_ptr), value :: ws
  integer(c_int64_t), value :: ws_size
  integer(c_int64_t) :: i, hit
  real(c_double) :: kr, first_d
  kr = real(k, c_double)
  out_index(1) = -1_8
  out_value(1) = -1.0d0
  if (len_1d <= 0_8) then
    if (.not. c_associated(ws) .or. ws_size <= 0_8) then
      do i = 0, 0
        cycle
      end do
    end if
    return
  end if
  first_d = huge(0.0d0)
  !$omp parallel do reduction(min: first_d) schedule(static)
  do i = 1, len_1d
    first_d = min(first_d, merge(real(i - 1_8, c_double), huge(0.0d0), a(i) > kr))
  end do
  !$omp end parallel do
  if (first_d < huge(0.0d0)) then
    hit = int(first_d, c_int64_t)
    out_index(1) = hit
    out_value(1) = a(hit + 1)
  end if
  if (c_associated(ws) .and. ws_size > 0_8) then
    do i = 0, 0
      cycle
    end do
  end if
end subroutine ext_break_capture_fp64
