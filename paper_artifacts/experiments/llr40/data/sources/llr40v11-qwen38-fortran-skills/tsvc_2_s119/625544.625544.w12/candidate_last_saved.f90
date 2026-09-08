subroutine tsvc_2_s119_fp64(aa, bb, len_2d) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer :: n, b, nb, s, ti, tj, tmin, tmax, p0, p1, q0, q1, p, q
  n = int(len_2d)
  b = 256
  nb = (n - 2 + b - 1) / b + 1
  do s = 0, 2*(nb-1)
    tmin = max(0, s - (nb - 1))
    tmax = min(nb - 1, s)
    !$omp do schedule(static)
    do ti = tmin, tmax
      tj = s - ti
      p0 = 2 + ti*b
      p1 = min(p0 + b - 1, n)
      q0 = 2 + tj*b
      q1 = min(q0 + b - 1, n)
      do q = q0, q1
        do p = p0, p1
          aa(p, q) = aa(p-1, q-1) + bb(p, q)
        end do
      end do
    end do
    !$omp end do
  end do
end subroutine tsvc_2_s119_fp64
