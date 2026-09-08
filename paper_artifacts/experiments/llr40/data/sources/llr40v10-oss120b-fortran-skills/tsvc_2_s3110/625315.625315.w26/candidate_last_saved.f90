subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C)
  use iso_c_binding, only: c_int64_t, c_double, c_int8_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(out) :: bb(1)
  integer(c_int8_t), dimension(*), intent(inout) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i, j
  integer(c_int64_t) :: xindex, yindex
  integer(c_int64_t) :: local_x, local_y
  real(c_double) :: maxv, local_max, v, chksum

  maxv = -huge(0.0_c_double)
  xindex = -1_c_int64_t
  yindex = -1_c_int64_t

  !$omp parallel private(i, j, v, local_max, local_x, local_y) shared(maxv, xindex, yindex)
    local_max = -huge(0.0_c_double)
    local_x = -1_c_int64_t
    local_y = -1_c_int64_t
    !$omp do schedule(static)
    do i = 1_c_int64_t, LEN_2D
      !$omp simd private(j, v)
      do j = 1_c_int64_t, LEN_2D
        v = aa(j, i)
        if (v > local_max) then
          local_max = v
          local_x = i - 1_c_int64_t
          local_y = j - 1_c_int64_t
        else if (v == local_max) then
          if ((i - 1_c_int64_t) * LEN_2D + (j - 1_c_int64_t) < local_x * LEN_2D + local_y) then
            local_x = i - 1_c_int64_t
            local_y = j - 1_c_int64_t
          end if
        end if
      end do
    end do
    !$omp end do
    !$omp critical
      if (local_max > maxv .or. (local_max == maxv .and. (local_x * LEN_2D + local_y) < (xindex * LEN_2D + yindex))) then
        maxv = local_max
        xindex = local_x
        yindex = local_y
      end if
    !$omp end critical
  !$omp end parallel

  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1) = chksum
end subroutine tsvc_2_s3110_fp64
