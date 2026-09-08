subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(out) :: bb(2, 2)

  integer(c_int64_t) :: i, j, xindex, yindex, first_idx, k
  real(c_double) :: maxv, chksum
  real(c_double) :: maxv_local
  integer(c_int64_t) :: idx_local

  maxv = -huge(0.0d0)
  first_idx = huge(1_c_int64_t)

  !$omp parallel private(maxv_local, idx_local, i, j, k) shared(maxv, first_idx)
  maxv_local = -huge(0.0d0)
  !$omp do schedule(static)
  do i = 1, LEN_2D
    !$omp simd reduction(max:maxv_local)
    do j = 1, LEN_2D
      if (aa(j, i) > maxv_local) maxv_local = aa(j, i)
    end do
  end do
  !$omp end do
  idx_local = huge(1_c_int64_t)
  !$omp do schedule(static)
  do i = 1, LEN_2D
    do j = 1, LEN_2D
      if (aa(j, i) == maxv_local) then
        k = (i - 1) * LEN_2D + (j - 1)
        if (k < idx_local) idx_local = k
      end if
    end do
  end do
  !$omp end do
  !$omp critical
  if (maxv_local > maxv .or. (maxv_local == maxv .and. idx_local < first_idx)) then
    maxv = maxv_local
    first_idx = idx_local
  end if
  !$omp end critical
  !$omp end parallel

  xindex = first_idx / LEN_2D
  yindex = mod(first_idx, LEN_2D)
  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1, 1) = chksum
end subroutine tsvc_2_s3110_fp64
