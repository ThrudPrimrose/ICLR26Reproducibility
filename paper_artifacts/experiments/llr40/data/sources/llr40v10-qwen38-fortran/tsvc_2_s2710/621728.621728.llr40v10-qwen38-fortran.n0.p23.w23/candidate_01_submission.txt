subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, nraw) bind(C, name='tsvc_2_s2710_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*), b(*), c(*)
  real(c_double), intent(in) :: d(*), e(*), x(*)
  integer(c_int64_t), intent(in), target :: nraw(*)
  integer(c_int64_t) :: n, i
  type(c_ptr) :: cp
  logical :: x0pos, m

  cp = c_loc(nraw(1))
  n = transfer(cp, 0_8)
  x0pos = x(1) > 0.0d0
  if (n > 10) then
    if (x0pos) then
      !$omp parallel do schedule(static)
      do i = 1, n
        m = a(i) > b(i)
        a(i) = a(i) + merge(b(i)*d(i), 0.0d0, m)
        b(i) = merge(b(i), a(i) + e(i)*e(i), m)
        c(i) = merge(c(i) + d(i)*d(i), a(i) + d(i)*d(i), m)
      end do
    else
      !$omp parallel do schedule(static)
      do i = 1, n
        m = a(i) > b(i)
        a(i) = a(i) + merge(b(i)*d(i), 0.0d0, m)
        b(i) = merge(b(i), a(i) + e(i)*e(i), m)
        c(i) = merge(c(i) + d(i)*d(i), c(i) + e(i)*e(i), m)
      end do
    end if
  else
    do i = 1, n
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i)*d(i)
        c(i) = d(i)*e(i) + 1.0d0
      else
        b(i) = a(i) + e(i)*e(i)
        if (x0pos) then
          c(i) = a(i) + d(i)*d(i)
        else
          c(i) = c(i) + e(i)*e(i)
        end if
      end if
    end do
  end if
end subroutine tsvc_2_s2710_fp64
