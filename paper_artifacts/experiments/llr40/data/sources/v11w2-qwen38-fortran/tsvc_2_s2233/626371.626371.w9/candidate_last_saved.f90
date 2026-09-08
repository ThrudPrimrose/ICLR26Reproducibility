! TSVC tsvc_2 s2233 -- two independent column scans.
!   aa[j,i] = aa[7,i] + sum_{t=8..j} cc[t,i]
!   bb[i,j] = bb[7,j] + sum_{t=8..i} cc[t,j]
! Work is partitioned into 8-column blocks; each row touch is one
! contiguous 8-double section per array.  Per-column accumulation
! order is unchanged from the reference (bitwise identical output).
subroutine tsvc_2_s2233_fp64(aa, bb, cc, len_2d, ws, ws_bytes) bind(C, name='tsvc_2_s2233_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(inout) :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), value    :: len_2d
  integer(c_int8_t), intent(inout) :: ws(*)
  integer(c_int64_t), value    :: ws_bytes

  integer :: n, r, blk, col, nb, nfull, w, k
  integer, parameter :: WB = 8
  real(c_double) :: s8(WB), t8(WB), x8(WB)

  if (len_2d <= 8) return
  n = int(len_2d)
  nfull = (n - 8) / WB
  nb = (n - 8 + WB - 1) / WB

  !$omp parallel do schedule(static) private(blk, r, col, k, s8, t8, x8)
  do blk = 0, nfull - 1
     col = 8 + WB * blk
     do k = 1, WB
        s8(k) = aa(7*n + col + k)
        t8(k) = bb(7*n + col + k)
     end do
     do r = 8, n - 1
        do k = 1, WB
           x8(k) = cc(r*n + col + k)
        end do
        do k = 1, WB
           s8(k) = s8(k) + x8(k)
           t8(k) = t8(k) + x8(k)
        end do
        do k = 1, WB
           aa(r*n + col + k) = s8(k)
           bb(r*n + col + k) = t8(k)
        end do
     end do
  end do
  !$omp end parallel do

  if (nfull < nb) then
     col = 8 + WB * nfull
     w = n - col
     do k = 1, w
        s8(k) = aa(7*n + col + k)
        t8(k) = bb(7*n + col + k)
     end do
     do r = 8, n - 1
        do k = 1, w
           x8(k) = cc(r*n + col + k)
        end do
        do k = 1, w
           s8(k) = s8(k) + x8(k)
           t8(k) = t8(k) + x8(k)
        end do
        do k = 1, w
           aa(r*n + col + k) = s8(k)
           bb(r*n + col + k) = t8(k)
        end do
     end do
  end if
end subroutine tsvc_2_s2233_fp64
