module tsvc_2_s119_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(c, name='tsvc_2_s119_fp64')
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: bb(*)
    integer(c_int64_t), value     :: LEN_2D
    integer(c_int64_t) :: n, i, j, s, bi, bj, nb, istart, iend, jstart, jend, bimin, bimax
    integer(c_int64_t), parameter :: bs = 64_c_int64_t

    n = LEN_2D
    if (n <= 1_c_int64_t) return

    if (n < 512_c_int64_t) then
      do i = 1_c_int64_t, n - 1_c_int64_t
        do j = 1_c_int64_t, n - 1_c_int64_t
          aa(i*n + j + 1_c_int64_t) = aa((i - 1_c_int64_t)*n + (j - 1_c_int64_t) + 1_c_int64_t) + bb(i*n + j + 1_c_int64_t)
        end do
      end do
      return
    end if

    nb = (n - 1_c_int64_t + bs - 1_c_int64_t) / bs

    !$omp parallel default(shared) private(bi, bj, istart, iend, jstart, jend, i, j)
    !$omp single
    do s = 2_c_int64_t, 2_c_int64_t * nb
      bimin = max(1_c_int64_t, s - nb)
      bimax = min(nb, s - 1_c_int64_t)
      !$omp taskloop private(bi, bj, istart, iend, jstart, jend, i, j) grainsize(1)
      do bi = bimin, bimax
        bj = s - bi
        istart = (bi - 1_c_int64_t) * bs + 1_c_int64_t
        iend   = min(istart + bs - 1_c_int64_t, n - 1_c_int64_t)
        jstart = (bj - 1_c_int64_t) * bs + 1_c_int64_t
        jend   = min(jstart + bs - 1_c_int64_t, n - 1_c_int64_t)
        do i = istart, iend
          do j = jstart, jend
            aa(i*n + j + 1_c_int64_t) = aa((i - 1_c_int64_t)*n + (j - 1_c_int64_t) + 1_c_int64_t) + bb(i*n + j + 1_c_int64_t)
          end do
        end do
      end do
      !$omp end taskloop
    end do
    !$omp end single
    !$omp end parallel
  end subroutine tsvc_2_s119_fp64
end module tsvc_2_s119_mod
