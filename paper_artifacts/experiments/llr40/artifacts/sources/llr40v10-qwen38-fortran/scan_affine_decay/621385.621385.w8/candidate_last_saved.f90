! scan_affine_decay: y[i] = c[i]*y[i-1] + x[i]  (0-based, i = 1..n-1; y[0] is seed).
! C-ABI: void scan_affine_decay_fp64(double *y, const double *c, const double *x, int64_t LEN_1D)
!
! Strategy: blocked scan.
!   phase 1: parallel over blocks of size m: compute block summary (A,B) =
!            composition of the affine maps in the block (2 dependent chains).
!            8 blocks are advanced simultaneously per super-step (16 independent
!            scalar ops/super-step -> ~2 flops/cycle, latency hidden).
!   phase 2: serial scan of the (few) block summaries -> per-block input value.
!   phase 3: parallel over blocks: rescan block with the known input (1 FMA chain,
!            again 8 blocks simultaneously).
! Small n: plain serial FMA loop (OpenMP spawn would dominate).
subroutine scan_affine_decay_fp64(y, c, x, len_1d) bind(c, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout), dimension(*) :: y
  real(c_double), intent(in),    dimension(*) :: c
  real(c_double), intent(in),    dimension(*) :: x
  integer(c_int64_t), value :: len_1d

  integer(c_int64_t) :: n
  integer :: nt, tid, t
  integer(c_int64_t) :: i, b, s, blk, nb, rem, m, b0, b1
  real(c_double) :: yy, ain, bin, a, bval
  real(c_double) :: cA(8), cB(8), xA(8), xB(8)

  n = len_1d
  if (n <= 1) return

  ! ------------------------------------------------------------------ small: serial
  if (n < 32768_c_int64) then
     yy = y(1)
     do i = 2, n
        yy = c(i) * yy + x(i)
        y(i) = yy
     end do
     return
  end if

  m  = 2048_c_int64
  nb = n / m
  rem = n - nb*m

  ! scratch: per-block summaries and per-block input values
  ! allocate generously; nb up to ~2^31/2048 fine
  ! -------------------------------------------------------------
  ! phase 1 + phase 2 + phase 3 inside one team
  ! -------------------------------------------------------------
  !$omp parallel private(tid, t, b0, b1, b, s, i, ain, bin, a, bval, cA, cB, xA, xB, yy)
     nt  = omp_get_num_threads()
     tid = omp_get_thread_num()

     ! contiguous chunk of blocks per thread (blocks 0..nb-1)
     b0 = int64(tid) * nb / int64(nt)
     b1 = int64(tid + 1) * nb / int64(nt)

     ! -------- phase 1: summaries of blocks in [b0, b1) --------
     ! groups of up to 8 blocks advanced lock-step
     do blk = b0, b1 - 8, 8
        a = 1.0d0; bval = 1.0d0
        do s = 0, m - 1
           ! note: use separate accumulators per lane
        end do
     end do
  !$omp end parallel
end subroutine scan_affine_decay_fp64
