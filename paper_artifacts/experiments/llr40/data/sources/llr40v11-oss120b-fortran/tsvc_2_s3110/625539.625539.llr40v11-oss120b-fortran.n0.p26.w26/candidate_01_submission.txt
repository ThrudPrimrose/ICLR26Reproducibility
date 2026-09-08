module tsvc_2_s3110_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
    ! Arguments: aa - input array (LEN_2D*LEN_2D), bb - output array (size >=1), LEN_2D - dimension
    implicit none
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(in) :: aa(*)
    real(c_double), intent(out) :: bb(*)
    integer(c_int64_t) :: i, j, idx
    real(c_double) :: maxv, v, chksum
    integer(c_int64_t) :: xindex, yindex
    integer(c_int64_t) :: local_max_i, local_max_j
    real(c_double) :: local_maxv
    ! Initialize with the first element
    maxv = aa(1)
    xindex = 0_c_int64_t
    yindex = 0_c_int64_t
    ! Parallel reduction
    !$omp parallel private(i,j,idx,v,local_maxv,local_max_i,local_max_j) shared(maxv,xindex,yindex)
      local_maxv = -huge(1.0_c_double)  ! Very small initial value
      local_max_i = 0_c_int64_t
      local_max_j = 0_c_int64_t
      !$omp do collapse(2) schedule(static)
      do i = 0_c_int64_t, LEN_2D-1_c_int64_t
        do j = 0_c_int64_t, LEN_2D-1_c_int64_t
          idx = i * LEN_2D + j
          v = aa(idx+1)  ! Fortran arrays are 1-indexed
          if (v > local_maxv) then
            local_maxv = v
            local_max_i = i
            local_max_j = j
          end if
        end do
      end do
      !$omp critical
        if (local_maxv > maxv) then
          maxv = local_maxv
          xindex = local_max_i
          yindex = local_max_j
        end if
      !$omp end critical
    !$omp end parallel
    chksum = maxv + real(xindex, kind=c_double) + real(yindex, kind=c_double)
    bb(1) = chksum
  end subroutine tsvc_2_s3110_fp64
end module tsvc_2_s3110_mod
