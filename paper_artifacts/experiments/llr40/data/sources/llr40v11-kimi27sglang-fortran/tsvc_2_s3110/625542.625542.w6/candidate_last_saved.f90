! TSVC s3110 hand-optimized Fortran implementation
subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(in) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  integer(c_int64_t), value, intent(in) :: LEN_2D

  integer(c_int64_t) :: n, i, j, xindex, yindex, linear
  real(c_double) :: maxv, v

  n = LEN_2D
  maxv = aa(1)
  xindex = 0_c_int64_t
  yindex = 0_c_int64_t

  !$omp parallel do default(none) schedule(static) &
  !$omp shared(aa, n) private(i, j, v) &
  !$omp reduction(max: maxv)
  do i = 0, n - 1
    do j = 1, n
      v = aa(i * n + j)
      if (v > maxv) then
        maxv = v
      end if
    end do
  end do
  !$omp end parallel do

  ! second pass to locate first (row-major) occurrence of maxv
  linear = huge(1_c_int64_t)
  !$omp parallel do default(none) schedule(static) &
  !$omp shared(aa, n, maxv) private(i, j, v) &
  !$omp reduction(min: linear)
  do i = 0, n - 1
    do j = 1, n
      v = aa(i * n + j)
      if (v == maxv) then
        if (i * n + j - 1 < linear) linear = i * n + j - 1
      end if
    end do
  end do
  !$omp end parallel do

  xindex = linear / n
  yindex = mod(linear, n)
  bb(1) = maxv + dble(xindex) + dble(yindex)
end subroutine tsvc_2_s3110_fp64
