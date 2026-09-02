subroutine wf_triangular_fp64(a, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: n, bs, nt, d, ti, tj, ilo, ihi, jlo, jhi, i, j, ti0, ti1, jb
  real(c_double) :: west, val

  n = LEN_2D
  if (n < 2) return

  bs = 64
  nt = (n + bs - 1) / bs

  !$omp parallel private(d, ti, tj, ilo, ihi, jlo, jhi, i, j, ti0, ti1, jb, west, val)
  do d = 2, 2*nt
    ti0 = max(1_c_int64_t, d - nt)
    ti1 = min(nt, d - 1_c_int64_t)
    ti1 = min(ti1, d / 2)
    if (ti0 <= ti1) then
      !$omp do schedule(static)
      do ti = ti0, ti1
        tj = d - ti
        ilo = (ti - 1) * bs + 1
        ihi = min(n, ti * bs)
        jlo = (tj - 1) * bs + 1
        jhi = min(n, tj * bs)
        do i = max(ilo, 2_c_int64_t), ihi
          jb = max(jlo, i)
          west = a(jb - 1, i)
          do j = jb, jhi
            val = a(j, i) + a(j, i - 1) + west
            a(j, i) = val
            west = val
          end do
        end do
      end do
      !$omp end do
    end if
  end do
  !$omp end parallel
end subroutine wf_triangular_fp64
