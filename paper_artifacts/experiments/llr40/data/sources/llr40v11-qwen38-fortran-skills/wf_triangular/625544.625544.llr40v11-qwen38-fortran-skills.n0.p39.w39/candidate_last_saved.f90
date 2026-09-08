subroutine wf_triangular_fp64(a, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)

  integer(c_int64_t) :: nb, d, bi, bj, bl, bh, ci0, ci1, cj0, cj1, q0, q1, p0, p1, q, p
  integer(c_int64_t), parameter :: B = 64_c_int64_t

  nb = (len_2d + B - 1) / B

  !$omp parallel default(none) shared(a, len_2d, nb) private(d, bi, bj, bl, bh, ci0, ci1, cj0, cj1, q0, q1, p0, p1, q, p)
  do d = 0, 2 * nb - 2
    bl = max(0_c_int64_t, d - (nb - 1_c_int64_t))
    bh = min(nb - 1_c_int64_t, d / 2_c_int64_t)
    !$omp do schedule(static)
    do bi = bl, bh
      bj = d - bi
      ci0 = max(bi * B, 1_c_int64_t)
      ci1 = min((bi + 1_c_int64_t) * B, len_2d) - 1_c_int64_t
      cj0 = bj * B
      cj1 = min((bj + 1_c_int64_t) * B, len_2d) - 1_c_int64_t
      if (ci0 <= ci1 .and. ci0 <= cj1) then
        q0 = ci0 + 1_c_int64_t
        q1 = ci1 + 1_c_int64_t
        do q = q0, q1
          p0 = max(cj0 + 1_c_int64_t, q)
          p1 = cj1 + 1_c_int64_t
          do p = p0, p1
            a(p, q) = a(p, q) + a(p, q - 1_c_int64_t) + a(p - 1_c_int64_t, q)
          end do
        end do
      end if
    end do
    !$omp end do
  end do
  !$omp end parallel

end subroutine wf_triangular_fp64
