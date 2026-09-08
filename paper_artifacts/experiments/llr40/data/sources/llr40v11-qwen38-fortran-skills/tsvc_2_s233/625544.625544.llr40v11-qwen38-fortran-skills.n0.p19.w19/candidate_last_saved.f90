subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: cc(len_2d, len_2d)

  integer(c_int64_t) :: i, j, istart, iend, lastfull, nbcol, k, idx
  integer(c_int64_t), parameter :: B = 8
  integer(c_int64_t), parameter :: PD = 8
  real(c_double) :: carry(B)
  real(c_double) :: ccbuf(PD, B)

  if (len_2d > 8) then

    ! ---- Loop 1: aa(i,j) = aa(i,j-1) + cc(i,j)  (scan over j, per column i)
    ! Carry the running value in a register (no aa re-read); software-prefetch
    ! the strided cc loads (stride n in j) PD steps ahead to hide DRAM latency.
    ! Blocks of B contiguous columns are independent -> parallel over blocks.
    nbcol = (len_2d - 8) / B
    lastfull = 9 + (nbcol - 1) * B
    !$omp parallel do private(carry, ccbuf, iend, idx)
    do istart = 9, lastfull, B
      iend = istart + B - 1
      carry(:) = aa(istart:iend, 8)
      do idx = 0, PD - 1
        ccbuf(idx + 1, :) = cc(istart:iend, 9 + idx)
      end do
      do j = 9, len_2d
        idx = mod(j - 9, PD) + 1
        carry(:) = carry(:) + ccbuf(idx, :)
        aa(istart:iend, j) = carry(:)
        if (j + PD <= len_2d) then
          ccbuf(idx, :) = cc(istart:iend, j + PD)
        end if
      end do
    end do
    ! ragged tail columns
    !$omp parallel do
    do k = 9 + nbcol * B, len_2d
      do j = 9, len_2d
        aa(k, j) = aa(k, j - 1) + cc(k, j)
      end do
    end do

    ! ---- Loop 2: bb(i,j) = bb(i-1,j) + cc(i,j)  (scan over i, per row j)
    ! i is unit stride; swap so j (rows) is the parallel outer loop.
    !$omp parallel do
    do j = 9, len_2d
      do i = 9, len_2d
        bb(i, j) = bb(i - 1, j) + cc(i, j)
      end do
    end do

  end if
end subroutine tsvc_2_s233_fp64
