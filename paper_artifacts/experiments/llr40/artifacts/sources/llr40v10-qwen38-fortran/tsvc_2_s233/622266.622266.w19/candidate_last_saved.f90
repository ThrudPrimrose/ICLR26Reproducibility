subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C, name='tsvc_2_s233_fp64')
  use iso_c_binding, only: c_int64_t
  implicit none
  double precision, intent(inout) :: aa(:), bb(:)
  double precision, intent(in)    :: cc(:)
  integer(c_int64_t), intent(in)  :: len_2d

  integer(c_int64_t), parameter :: BW = 32
  integer(c_int64_t) :: i, j, ib, k, n, nk
  double precision :: s(BW)

  n = len_2d
  if (n <= 8) return

  ! Part 1: per-column scan aa[j,i] = aa[j-1,i] + cc[j,i], i = 8..n-1.
  ! Blocks of 32 consecutive columns scanned in lockstep (contiguous 256B lines),
  ! 4 independent 256-bit chains -> vectorized, memory-level parallel.
!$omp parallel do schedule(static)
  do ib = 8, n-1, BW
    nk = min(BW, n - 8 - ib + 1)
    do k = 1, nk
      s(k) = aa(7*n + ib + k)
    end do
    if (nk == BW) then
      do j = 8, n-1
        do k = 1, BW
          s(k) = s(k) + cc(j*n + ib + k)
          aa(j*n + ib + k) = s(k)
        end do
      end do
    else
      do j = 8, n-1
        do k = 1, nk
          s(k) = s(k) + cc(j*n + ib + k)
          aa(j*n + ib + k) = s(k)
        end do
      end do
    end if
  end do
!$omp end parallel do

  ! Part 2: per-row scan bb[j,i] = bb[j,i-1] + cc[j,i], j = 8..n-1 (contiguous).
!$omp parallel do schedule(static)
  do j = 8, n-1
    s(1) = bb(j*n + 8)
    do i = 8, n-1
      s(1) = s(1) + cc(j*n + i + 1)
      bb(j*n + i + 1) = s(1)
    end do
  end do
!$omp end parallel do

end subroutine
