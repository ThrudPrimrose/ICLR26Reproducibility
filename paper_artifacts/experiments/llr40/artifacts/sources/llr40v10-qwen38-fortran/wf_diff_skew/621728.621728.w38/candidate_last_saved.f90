subroutine wf_diff_skew_fp64(a, len_2d) bind(C, name='wf_diff_skew_fp64')
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(*)
  integer(c_int64_t) :: n, i, j, r0, r1, c0, c1
  integer :: iblk, jblk, sb, tb, dd, j0, j1, ib0

  n = len_2d
  if (n < 2) return

  if (n >= 4096) then
    sb = 64
    jblk = 8
  else
    sb = 32
    jblk = 4
  end if
  tb = int((n - 1 + jblk - 1) / jblk)
  iblk = int((n - 1 + sb - 1) / sb)

  ! block (0,0) has no dependencies: do it first
  do i = 1, min(int64(sb), n - 1)
    do j = 0, min(int64(tb) - 1, n - 2)
      a(i*n + j + 1) = a(i*n + j + 1) + a((i-1)*n + j + 1) + a((i-1)*n + j + 2)
    end do
  end do

  ! Wavefront: block (I,J) depends on blocks (I-1,J) and (I-1,J+1),
  ! both on the previous diagonal d-1 = I+J-1.
  ! The implicit barrier at the end of each parallel do guarantees
  ! that the whole diagonal d-1 is complete before diagonal d starts.
  !$omp team
  do dd = 1, iblk + jblk - 2
    j0 = max(0, dd - iblk + 1)
    j1 = min(dd, jblk - 1)
    !$omp parallel do
    do j = j0, j1
      ib0 = dd - j
      r0 = int64(ib0)*int64(sb) + 1
      r1 = min(int64(ib0 + 1)*int64(sb), n - 1)
      c0 = int64(j)*int64(tb)
      c1 = min(int64(j + 1)*int64(tb) - 1, n - 2)
      do i = r0, r1
        do j = c0, c1
          a(i*n + j + 1) = a(i*n + j + 1) + a((i-1)*n + j + 1) + a((i-1)*n + j + 2)
        end do
      end do
    end do
    !$omp end parallel do
  end do
  !$omp end team
end subroutine
