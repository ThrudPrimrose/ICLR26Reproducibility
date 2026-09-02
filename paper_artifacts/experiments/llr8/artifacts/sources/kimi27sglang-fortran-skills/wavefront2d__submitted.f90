subroutine wavefront2d_fp64(a, n) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int), parameter :: U = 8
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n,n)
  integer(c_int64_t) :: i, j, bi, bj, diag
  integer(c_int64_t) :: istart, iend, jstart, jend, jfirst, i0
  integer(c_int64_t) :: tile, ntiles
  integer(c_int) :: r
  real(c_double) :: cur(0:U-1), w(0:U-1), oldw(0:U-1), newv(0:U-1)
  real(c_double) :: north0, nw0

  if (n <= 2) return

  if (n < 128_c_int64_t) then
    do i = 2_c_int64_t, n
      w(0) = a(1_c_int64_t, i)
      nw0 = a(1_c_int64_t, i - 1_c_int64_t)
      do j = 2_c_int64_t, n
        cur(0) = a(j, i)
        north0 = a(j, i - 1_c_int64_t)
        newv(0) = 0.25d0 * (cur(0) + north0 + w(0) + nw0)
        a(j, i) = newv(0)
        nw0 = north0
        w(0) = newv(0)
      end do
    end do
    return
  end if

  tile = 128_c_int64_t
  ntiles = (n + tile - 1_c_int64_t) / tile

  !$omp parallel private(i, j, bi, bj, diag, istart, iend, jstart, jend, jfirst, i0, &
  !$omp&                  r, cur, w, oldw, newv, north0, nw0) &
  !$omp&          shared(tile, ntiles)
  do diag = 2_c_int64_t, 2_c_int64_t * ntiles
    !$omp do schedule(static)
    do bi = max(1_c_int64_t, diag - ntiles), min(ntiles, diag - 1_c_int64_t)
      bj = diag - bi
      istart = (bi - 1_c_int64_t) * tile + 1_c_int64_t
      iend = min(bi * tile, n)
      jstart = (bj - 1_c_int64_t) * tile + 1_c_int64_t
      jend = min(bj * tile, n)
      i0 = max(istart, 2_c_int64_t)
      jfirst = max(jstart, 2_c_int64_t)

      do i = i0, iend - int(U - 1, c_int64_t), int(U, c_int64_t)
        do r = 0, U - 1
          w(r) = a(jfirst - 1_c_int64_t, i + int(r, c_int64_t))
        end do
        nw0 = a(jfirst - 1_c_int64_t, i - 1_c_int64_t)
        do j = jfirst, jend
          do r = 0, U - 1
            cur(r) = a(j, i + int(r, c_int64_t))
          end do
          north0 = a(j, i - 1_c_int64_t)

          oldw(0) = w(0)
          newv(0) = 0.25d0 * (cur(0) + north0 + w(0) + nw0)
          a(j, i) = newv(0)
          do r = 1, U - 1
            oldw(r) = w(r)
            newv(r) = 0.25d0 * (cur(r) + newv(r - 1) + w(r) + oldw(r - 1))
            a(j, i + int(r, c_int64_t)) = newv(r)
          end do

          do r = 0, U - 1
            w(r) = newv(r)
          end do
          nw0 = north0
        end do
      end do

      i = i0 + int(U, c_int64_t) * ((iend - i0 + 1_c_int64_t) / int(U, c_int64_t))
      do while (i <= iend)
        w(0) = a(jfirst - 1_c_int64_t, i)
        nw0 = a(jfirst - 1_c_int64_t, i - 1_c_int64_t)
        do j = jfirst, jend
          cur(0) = a(j, i)
          north0 = a(j, i - 1_c_int64_t)
          newv(0) = 0.25d0 * (cur(0) + north0 + w(0) + nw0)
          a(j, i) = newv(0)
          nw0 = north0
          w(0) = newv(0)
        end do
        i = i + 1_c_int64_t
      end do
    end do
    !$omp end do
  end do
  !$omp end parallel
end subroutine wavefront2d_fp64
