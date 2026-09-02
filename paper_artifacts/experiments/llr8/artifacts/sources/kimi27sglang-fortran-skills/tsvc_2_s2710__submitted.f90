subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)

  integer :: i
  real(c_double) :: ai, bi, ci, di, ei
  logical(c_bool) :: gt

  if (LEN_1D > 8192_c_int64_t) then
    if (LEN_1D > 10_c_int64_t) then
      if (x(1) > 0.0_c_double) then
        !$omp parallel do simd
        do i = 1, int(LEN_1D)
          ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
          gt = ai > bi
          a(i) = merge(ai + bi * di, ai, gt)
          b(i) = merge(bi, ai + ei * ei, gt)
          c(i) = merge(ci + di * di, ai + di * di, gt)
        end do
        !$omp end parallel do simd
      else
        !$omp parallel do simd
        do i = 1, int(LEN_1D)
          ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
          gt = ai > bi
          a(i) = merge(ai + bi * di, ai, gt)
          b(i) = merge(bi, ai + ei * ei, gt)
          c(i) = merge(ci + di * di, ci + ei * ei, gt)
        end do
        !$omp end parallel do simd
      end if
    else
      if (x(1) > 0.0_c_double) then
        !$omp parallel do simd
        do i = 1, int(LEN_1D)
          ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
          gt = ai > bi
          a(i) = merge(ai + bi * di, ai, gt)
          b(i) = merge(bi, ai + ei * ei, gt)
          c(i) = merge(di * ei + 1.0_c_double, ai + di * di, gt)
        end do
        !$omp end parallel do simd
      else
        !$omp parallel do simd
        do i = 1, int(LEN_1D)
          ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
          gt = ai > bi
          a(i) = merge(ai + bi * di, ai, gt)
          b(i) = merge(bi, ai + ei * ei, gt)
          c(i) = merge(di * ei + 1.0_c_double, ci + ei * ei, gt)
        end do
        !$omp end parallel do simd
      end if
    end if
  else
    if (LEN_1D > 10_c_int64_t) then
      if (x(1) > 0.0_c_double) then
        do i = 1, int(LEN_1D)
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = c(i) + d(i) * d(i)
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = a(i) + d(i) * d(i)
          end if
        end do
      else
        do i = 1, int(LEN_1D)
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = c(i) + d(i) * d(i)
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = c(i) + e(i) * e(i)
          end if
        end do
      end if
    else
      if (x(1) > 0.0_c_double) then
        do i = 1, int(LEN_1D)
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = d(i) * e(i) + 1.0_c_double
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = a(i) + d(i) * d(i)
          end if
        end do
      else
        do i = 1, int(LEN_1D)
          if (a(i) > b(i)) then
            a(i) = a(i) + b(i) * d(i)
            c(i) = d(i) * e(i) + 1.0_c_double
          else
            b(i) = a(i) + e(i) * e(i)
            c(i) = c(i) + e(i) * e(i)
          end if
        end do
      end if
    end if
  end if

end subroutine tsvc_2_s2710_fp64
