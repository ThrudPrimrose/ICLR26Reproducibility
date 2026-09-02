subroutine wf_north_west_fp64(a, LEN_2D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(LEN_2D, LEN_2D)
  integer(c_int64_t) :: n, tile, ntiles, diag, bi, bj
  integer(c_int64_t) :: istart, iend, jstart, jend, i, j, jfirst
  real(c_double) :: west, cur, north, newv

  n = LEN_2D
  if (n < 2_c_int64_t) return

  if (n <= 128_c_int64_t) then
     do i = 2_c_int64_t, n
        do j = 2_c_int64_t, n
           a(j, i) = a(j, i) + a(j - 1_c_int64_t, i) + a(j, i - 1_c_int64_t)
        end do
     end do
     return
  end if

  tile = 64_c_int64_t
  ntiles = (n + tile - 1_c_int64_t) / tile

  !$omp parallel private(diag, bi, bj, istart, iend, jstart, jend, &
  !$omp&                   i, j, jfirst, west, cur, north, newv)
  do diag = 2_c_int64_t, 2_c_int64_t * ntiles
     !$omp do schedule(static)
     do bi = max(1_c_int64_t, diag - ntiles), min(ntiles, diag - 1_c_int64_t)
        bj = diag - bi
        istart = (bi - 1_c_int64_t) * tile + 1_c_int64_t
        iend   = min(bi * tile, n)
        jstart = (bj - 1_c_int64_t) * tile + 1_c_int64_t
        jend   = min(bj * tile, n)
        do i = max(istart, 2_c_int64_t), iend
           jfirst = max(jstart, 2_c_int64_t)
           west = a(jfirst - 1_c_int64_t, i)
           do j = jfirst, jend
              cur   = a(j, i)
              north = a(j, i - 1_c_int64_t)
              newv  = cur + north + west
              a(j, i) = newv
              west  = newv
           end do
        end do
     end do
     !$omp end do
  end do
  !$omp end parallel
end subroutine wf_north_west_fp64
