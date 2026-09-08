subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(c)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(in) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(out) :: bb(0:*)

  integer(c_int64_t) :: i, j, global_xi, global_yi, local_xi, local_yi
  real(c_double) :: global_maxv, local_maxv, row_max, chksum

  global_maxv = aa(0, 0)
  global_xi = 0
  global_yi = 0

  !$omp parallel private(i, j, row_max, local_maxv, local_xi, local_yi)
  local_maxv = -huge(1.0_c_double)
  local_xi = 0_c_int64_t
  local_yi = 0_c_int64_t
  !$omp do schedule(static)
  do i = 0, LEN_2D - 1
    row_max = maxval(aa(:, i))
    j = findloc(aa(:, i), value=row_max, dim=1) - 1_c_int64_t
    if (row_max > local_maxv) then
      local_maxv = row_max
      local_xi = i
      local_yi = j
    else if (row_max == local_maxv) then
      if (i < local_xi .or. (i == local_xi .and. j < local_yi)) then
        local_xi = i
        local_yi = j
      end if
    end if
  end do
  !$omp end do
  !$omp critical
  if (local_maxv > global_maxv) then
    global_maxv = local_maxv
    global_xi = local_xi
    global_yi = local_yi
  else if (local_maxv == global_maxv) then
    if (local_xi < global_xi .or. (local_xi == global_xi .and. local_yi < global_yi)) then
      global_xi = local_xi
      global_yi = local_yi
    end if
  end if
  !$omp end critical
  !$omp end parallel

  chksum = global_maxv + dble(global_xi) + dble(global_yi)
  bb(0) = chksum
end subroutine tsvc_2_s3110_fp64
