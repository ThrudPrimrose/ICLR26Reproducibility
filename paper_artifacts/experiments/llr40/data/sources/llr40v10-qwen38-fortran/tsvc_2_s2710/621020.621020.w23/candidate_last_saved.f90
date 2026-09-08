subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(C, name='tsvc_2_s2710_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout), contiguous :: a(:), b(:), c(:)
  real(c_double), intent(in),    contiguous :: d(:), e(:), x(:)
  integer(c_int64_t), value              :: len_1d
  integer  :: n, i
  logical  :: big, xpos

  n = int(len_1d)
  if (n <= 0) return
  big  = len_1d > 10
  xpos = x(1) > 0.0d0

  !$omp parallel do default(none) shared(a, b, c, d, e, big, xpos, n) schedule(static)
  do i = 1, n
    if (a(i) > b(i)) then
      a(i) = a(i) + b(i)*d(i)
      c(i) = merge(c(i) + d(i)*d(i), d(i)*e(i) + 1.0d0, big)
    else
      b(i) = a(i) + e(i)*e(i)
      c(i) = merge(a(i) + d(i)*d(i), c(i) + e(i)*e(i), xpos)
    end if
  end do
end subroutine tsvc_2_s2710_fp64
