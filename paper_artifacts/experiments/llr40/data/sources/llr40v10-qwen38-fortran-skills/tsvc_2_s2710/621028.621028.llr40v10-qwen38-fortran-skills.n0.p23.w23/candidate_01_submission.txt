subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(inout) :: c(len_1d)
  real(c_double), intent(in)    :: d(len_1d)
  real(c_double), intent(in)    :: e(len_1d)
  real(c_double), intent(in)    :: x(len_1d)
  integer(c_int64_t) :: i
  real(c_double) :: ai, bi, ci, di, ei
  logical :: t

  if (len_1d == 0) return

  if (len_1d > 10) then
    if (x(1) > 0.0d0) then
      !$omp parallel do simd
      do i = 1, len_1d
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        t = ai > bi
        a(i) = merge(ai + bi*di, ai, t)
        b(i) = merge(bi, ai + ei*ei, t)
        c(i) = merge(ci + di*di, ai + di*di, t)
      end do
    else
      !$omp parallel do simd
      do i = 1, len_1d
        ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
        t = ai > bi
        a(i) = merge(ai + bi*di, ai, t)
        b(i) = merge(bi, ai + ei*ei, t)
        c(i) = merge(ci + di*di, ci + ei*ei, t)
      end do
    end if
  else
    if (x(1) > 0.0d0) then
      do i = 1, len_1d
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = d(i) * e(i) + 1.0d0
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = a(i) + d(i) * d(i)
        end if
      end do
    else
      do i = 1, len_1d
        if (a(i) > b(i)) then
          a(i) = a(i) + b(i) * d(i)
          c(i) = d(i) * e(i) + 1.0d0
        else
          b(i) = a(i) + e(i) * e(i)
          c(i) = c(i) + e(i) * e(i)
        end if
      end do
    end if
  end if
end subroutine tsvc_2_s2710_fp64
