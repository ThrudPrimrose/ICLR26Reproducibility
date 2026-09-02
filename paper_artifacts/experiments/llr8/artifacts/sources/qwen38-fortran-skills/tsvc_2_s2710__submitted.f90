subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d), b(len_1d), c(len_1d)
  real(c_double), intent(inout) :: d(len_1d), e(len_1d)
  real(c_double), intent(in) :: x(len_1d)
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: n64, j
  integer :: n, i
  logical :: xpos
  real(c_double) :: aa, bb, cc, dd, ee
  logical :: m

  n64 = len_1d
  if (n64 > 2147480000) then
    ! pathologically large: simple correct serial fallback
    do j = 1, n64
      if (a(j) > b(j)) then
        a(j) = a(j) + b(j)*d(j)
        if (n64 > 10) then
          c(j) = c(j) + d(j)*d(j)
        else
          c(j) = d(j)*e(j) + 1.0d0
        end if
      else
        b(j) = a(j) + e(j)*e(j)
        if (x(1) > 0.0d0) then
          c(j) = a(j) + d(j)*d(j)
        else
          c(j) = c(j) + e(j)*e(j)
        end if
      end if
    end do
    return
  end if
  n = int(n64)
  if (n <= 10) then
    do i = 1, n
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i)*d(i)
        c(i) = d(i)*e(i) + 1.0d0
      else
        b(i) = a(i) + e(i)*e(i)
        if (x(1) > 0.0d0) then
          c(i) = a(i) + d(i)*d(i)
        else
          c(i) = c(i) + e(i)*e(i)
        end if
      end if
    end do
    return
  end if
  xpos = x(1) > 0.0d0
  if (xpos) then
    !$omp parallel do simd schedule(static)
    do i = 1, n
      aa = a(i); bb = b(i); cc = c(i); dd = d(i); ee = e(i)
      m = aa > bb
      a(i) = merge(aa + bb*dd, aa, m)
      b(i) = merge(bb, aa + ee*ee, m)
      c(i) = merge(cc + dd*dd, aa + dd*dd, m)
    end do
  else
    !$omp parallel do simd schedule(static)
    do i = 1, n
      aa = a(i); bb = b(i); cc = c(i); dd = d(i); ee = e(i)
      m = aa > bb
      a(i) = merge(aa + bb*dd, aa, m)
      b(i) = merge(bb, aa + ee*ee, m)
      c(i) = cc + merge(dd*dd, ee*ee, m)
    end do
  end if
end subroutine tsvc_2_s2710_fp64
