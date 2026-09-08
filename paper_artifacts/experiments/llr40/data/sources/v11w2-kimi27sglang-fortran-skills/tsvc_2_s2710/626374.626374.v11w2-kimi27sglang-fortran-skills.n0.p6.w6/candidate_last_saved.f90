subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(C, name='tsvc_2_s2710_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d), b(len_1d), c(len_1d)
  real(c_double), intent(in) :: d(len_1d), e(len_1d), x(len_1d)
  integer :: i, n
  real(c_double) :: m, nm, lenm, nlenm, xm, nxm, ai, bi, di, ei, ct, cf

  n = int(len_1d)
  lenm = merge(1.0d0, 0.0d0, len_1d > 10)
  nlenm = 1.0d0 - lenm
  xm = merge(1.0d0, 0.0d0, x(1) > 0.0d0)
  nxm = 1.0d0 - xm

  do concurrent (i = 1:n) local(m, nm, ai, bi, di, ei, ct, cf)
     ai = a(i); bi = b(i); di = d(i); ei = e(i)
     m = merge(1.0d0, 0.0d0, ai > bi)
     nm = 1.0d0 - m
     ct = lenm*(c(i) + di*di) + nlenm*(di*ei + 1.0d0)
     cf = xm*(ai + di*di) + nxm*(c(i) + ei*ei)
     c(i) = m*ct + nm*cf
     b(i) = m*bi + nm*(ai + ei*ei)
     a(i) = ai + m*bi*di
  end do
end subroutine tsvc_2_s2710_fp64
