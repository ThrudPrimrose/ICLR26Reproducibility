subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
  real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)
  integer(c_int64_t) :: i
  logical :: big_len, xpos
  big_len = LEN_1D > 10_c_int64_t
  xpos = x(1) > 0.0_c_double
  !$omp parallel do default(none) schedule(static) if(LEN_1D > 1000) shared(a,b,c,d,e,x,LEN_1D,big_len,xpos) private(i)
  do i = 1, LEN_1D
    if (a(i) > b(i)) then
      a(i) = a(i) + b(i) * d(i)
      if (big_len) then
        c(i) = c(i) + d(i) * d(i)
      else
        c(i) = d(i) * e(i) + 1.0_c_double
      end if
    else
      b(i) = a(i) + e(i) * e(i)
      if (xpos) then
        c(i) = a(i) + d(i) * d(i)
      else
        c(i) = c(i) + e(i) * e(i)
      end if
    end if
  end do
  !$omp end parallel do
end subroutine tsvc_2_s2710_fp64
