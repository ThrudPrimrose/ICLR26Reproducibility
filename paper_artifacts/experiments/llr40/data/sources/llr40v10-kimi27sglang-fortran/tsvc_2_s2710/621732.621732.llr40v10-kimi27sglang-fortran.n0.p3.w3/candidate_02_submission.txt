subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C, name="tsvc_2_s2710_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: ai, bi, ci, di, ei
  real(c_double) :: at, bt, ct_t, ct_f
  logical :: cond, big, xpos

  big = LEN_1D > 10_c_int64_t
  xpos = x(1) > 0.0_c_double

  do i = 1, LEN_1D
    ai = a(i)
    bi = b(i)
    ci = c(i)
    di = d(i)
    ei = e(i)

    cond = ai > bi

    at = ai + bi * di
    bt = ai + ei * ei

    if (big) then
      ct_t = ci + di * di
    else
      ct_t = di * ei + 1.0_c_double
    end if

    if (xpos) then
      ct_f = ai + di * di
    else
      ct_f = ci + ei * ei
    end if

    a(i) = merge(at, ai, cond)
    b(i) = merge(bi, bt, cond)
    c(i) = merge(ct_t, ct_f, cond)
  end do
end subroutine tsvc_2_s2710_fp64
