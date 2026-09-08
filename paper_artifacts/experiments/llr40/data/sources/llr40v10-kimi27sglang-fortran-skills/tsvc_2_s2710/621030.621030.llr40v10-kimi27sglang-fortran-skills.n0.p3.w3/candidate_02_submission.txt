subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i
  real(c_double) :: ai, bi, ci, di, ei
  logical :: cond, len_gt_10, x_pos

  len_gt_10 = LEN_1D > 10
  x_pos = x(1) > 0.0d0

  if (len_gt_10) then
    if (x_pos) then
      !$omp parallel do simd
      do i = 1, LEN_1D
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        cond = ai > bi
        a(i) = merge(ai + bi * di, ai, cond)
        b(i) = merge(bi, ai + ei * ei, cond)
        c(i) = merge(ci + di * di, ai + di * di, cond)
      end do
      !$omp end parallel do simd
    else
      !$omp parallel do simd
      do i = 1, LEN_1D
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        cond = ai > bi
        a(i) = merge(ai + bi * di, ai, cond)
        b(i) = merge(bi, ai + ei * ei, cond)
        c(i) = merge(ci + di * di, ci + ei * ei, cond)
      end do
      !$omp end parallel do simd
    end if
  else
    if (x_pos) then
      !$omp parallel do simd
      do i = 1, LEN_1D
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        cond = ai > bi
        a(i) = merge(ai + bi * di, ai, cond)
        b(i) = merge(bi, ai + ei * ei, cond)
        c(i) = merge(di * ei + 1.0d0, ai + di * di, cond)
      end do
      !$omp end parallel do simd
    else
      !$omp parallel do simd
      do i = 1, LEN_1D
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        cond = ai > bi
        a(i) = merge(ai + bi * di, ai, cond)
        b(i) = merge(bi, ai + ei * ei, cond)
        c(i) = merge(di * ei + 1.0d0, ci + ei * ei, cond)
      end do
      !$omp end parallel do simd
    end if
  end if
end subroutine tsvc_2_s2710_fp64
