subroutine wavefront2d_fp64(a, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer :: n, b, nt, s, umin, umax, u, v
  integer :: ilow, ihigh, jlow, jhigh, t, tmax, i, j, ilo, ihi
  n = int(len_2d)
  if (n <= 1) return
  b = 32
  nt = (n - 1 + b - 1) / b
  do s = 0, 2 * nt - 2
    umin = max(0, s - nt + 1)
    umax = min(nt - 1, s)
    !$omp parallel do schedule(static)
    do u = umin, umax
      v = s - u
      ilow = u * b + 2
      ihigh = min((u + 1) * b + 1, n)
      jlow = v * b + 2
      jhigh = min((v + 1) * b + 1, n)
      tmax = (ihigh - ilow) + (jhigh - jlow)
      do t = 0, tmax
        ilo = max(ilow, t + ilow - (jhigh - jlow))
        ihi = min(ihigh, t + ilow)
        do i = ilo, ihi
          j = t - i + ilow + jlow
          a(i, j) = 0.25d0 * (a(i, j) + a(i-1, j) + a(i, j-1) + a(i-1, j-1))
        end do
      end do
    end do
  end do
  if (workspace_size > 0) workspace(1) = workspace(1)
end subroutine wavefront2d_fp64
