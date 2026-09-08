subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(C)
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double) :: bprev, c0, d0, e0, c1, d1, e1

  bprev = b(1)
  do i = 2, LEN_1D - 1, 2
    c0 = c(i)
    d0 = d(i)
    e0 = e(i)
    a(i) = bprev + c0 * d0
    bprev = a(i) + c0 * e0
    b(i) = bprev

    c1 = c(i + 1)
    d1 = d(i + 1)
    e1 = e(i + 1)
    a(i + 1) = bprev + c1 * d1
    bprev = a(i + 1) + c1 * e1
    b(i + 1) = bprev
  end do
  if (i == LEN_1D) then
    c0 = c(i)
    d0 = d(i)
    e0 = e(i)
    a(i) = bprev + c0 * d0
    bprev = a(i) + c0 * e0
    b(i) = bprev
  end if
end subroutine tsvc_2_s323_fp64
