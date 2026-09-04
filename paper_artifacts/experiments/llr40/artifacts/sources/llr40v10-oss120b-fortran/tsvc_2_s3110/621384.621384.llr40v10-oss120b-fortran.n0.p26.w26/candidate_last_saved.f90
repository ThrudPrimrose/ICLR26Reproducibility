module tsvc_2_s3110_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
    ! Arguments: aa is input double array of size LEN_2D*LEN_2D, bb is output double array (at least size 1)
    ! LEN_2D is the dimension of the square matrix.
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(in) :: aa(*)
    real(c_double), intent(out) :: bb(*)

    integer(c_int64_t) :: i, j, idx_lin, best_idx
    real(c_double) :: maxv, v, chksum
    integer(c_int64_t) :: xindex, yindex
    real(c_double) :: local_max
    integer(c_int64_t) :: local_idx

    maxv = -huge(0.0_c_double)
    best_idx = -1_c_int64_t

    !$omp parallel default(none) shared(aa, LEN_2D, maxv, best_idx) private(i, j, idx_lin, v, local_max, local_idx)
      local_max = -huge(0.0_c_double)
      local_idx = -1_c_int64_t
      !$omp do schedule(static)
      do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
        do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
          idx_lin = i * LEN_2D + j
          v = aa(idx_lin + 1_c_int64_t)
          if (v > local_max) then
            local_max = v
            local_idx = idx_lin
          else if (v == local_max .and. idx_lin < local_idx) then
            local_idx = idx_lin
          end if
        end do
      end do
      !$omp critical
      if (local_max > maxv) then
        maxv = local_max
        best_idx = local_idx
      else if (local_max == maxv .and. local_idx < best_idx) then
        best_idx = local_idx
      end if
      !$omp end critical
    !$omp end parallel

    if (best_idx < 0_c_int64_t) then
      xindex = 0_c_int64_t
      yindex = 0_c_int64_t
    else
      xindex = best_idx / LEN_2D
      yindex = mod(best_idx, LEN_2D)
    end if

    chksum = maxv + real(xindex, c_double) + real(yindex, c_double)
    bb(1) = chksum
  end subroutine tsvc_2_s3110_fp64
end module tsvc_2_s3110_mod
