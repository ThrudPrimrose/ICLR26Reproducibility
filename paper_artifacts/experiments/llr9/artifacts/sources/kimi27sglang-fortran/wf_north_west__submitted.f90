subroutine wf_north_west_fp64(a, n, workspace, wslen) bind(c, name='wf_north_west_fp64')
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double, c_int8_t
  implicit none
  integer(c_int64_t), value :: n
  integer(c_int8_t), intent(in) :: workspace(*)
  integer(c_int64_t), value :: wslen
  real(c_double), intent(inout) :: a(n, n)
  integer(c_int64_t) :: i, j, s, bi, bj, bd, nb, i1, i2, j1, j2, nloc
  integer(c_int64_t), parameter :: B = 64
  nloc = n
  if (nloc < 256) then
    do i = 2, nloc
      do j = 2, nloc
        a(j, i) = a(j, i) + a(j, i-1) + a(j-1, i)
      end do
    end do
  else
    nb = (nloc + B - 1) / B
    !$omp parallel private(i, j, s, bi, bj, bd, i1, i2, j1, j2)
    do bd = 2, 2*nb
      bj = max(1_c_int64_t, bd - nb)
      s = min(nb, bd - 1_c_int64_t) - bj + 1
      !$omp do schedule(static)
      do bi = 0, s - 1
        bj = max(1_c_int64_t, bd - nb) + bi
        i1 = (bd - bj - 1) * B + 1
        i2 = min((bd - bj) * B, nloc)
        j1 = (bj - 1) * B + 1
        j2 = min(bj * B, nloc)
        do i = max(i1, 2_c_int64_t), i2
          do j = max(j1, 2_c_int64_t), j2
            a(j, i) = a(j, i) + a(j, i-1) + a(j-1, i)
          end do
        end do
      end do
      !$omp end do
    end do
    !$omp end parallel
  end if
end subroutine wf_north_west_fp64
