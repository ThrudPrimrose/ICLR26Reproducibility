subroutine tsvc_2_s231_fp64(aa, bb, n) bind(C, name='tsvc_2_s231_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: n
  real(c_double), intent(inout) :: aa(n, n)
  real(c_double), intent(in)    :: bb(n, n)
  integer(c_int64_t) :: g, i, v, n64
  real(c_double) :: p(64), t(64)

  if (n < 2) return
  n64 = (n / 64) * 64
  if (n64 > 0) then
  !$omp parallel do schedule(static) private(p, t)
  do g = 1, n64, 64
    p = aa(g : g + 63, 1)
    do v = 2, n
      t = p + bb(g : g + 63, v)
      aa(g : g + 63, v) = t
      p = t
    end do
  end do
  end if
  if (n64 < n) then
  !$omp parallel do schedule(static)
  do i = n64 + 1, n
    do v = 2, n
      aa(i, v) = aa(i, v - 1) + bb(i, v)
    end do
  end do
  end if
end subroutine tsvc_2_s231_fp64
