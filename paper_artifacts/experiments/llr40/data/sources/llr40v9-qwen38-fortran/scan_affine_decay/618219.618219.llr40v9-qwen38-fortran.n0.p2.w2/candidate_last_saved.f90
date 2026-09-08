! Parallel segment-pipelined blocked scan for y(i) = c(i)*y(i-1) + x(i), i = 2..n (1-based).
! Element 1 is the seed. The recurrence is a scan over the non-commutative monoid of
! affine maps f_i(t) = c(i)*t + x(i); composing (a,b) after (A,B): (A*a, A*b + B).
!
! The array is processed in SEG segments. Within each segment:
!   P1 (parallel): per-block endpoint map via Kogge-Stone chunks of 32.
!   P2 (serial)  : exclusive scan over the ~2K block maps -> carry per block.
!   P3 (parallel): per-element prefix via KS chunks; y = A*carry + B.
! Segments are small enough that c+x of one segment stays in L3, so P3's re-read
! is an L3 hit and DRAM sees c, x once and y once.
subroutine scan_affine_decay_fp64(pc, px, py, LEN_1D) bind(C, name="scan_affine_decay_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value :: pc, px, py
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), pointer :: c(:), x(:), y(:)

  integer(c_int64_t) :: n, nb, b, j, lo, hi, s, d, m, ch, segstart, segend, segn
  integer(c_int64_t), parameter :: BLK  = 4096_8     ! elements per block
  integer(c_int64_t), parameter :: SEG  = 2_8**22    ! elements per segment (128 MB of c+x)
  integer(c_int64_t), parameter :: I2   = 2_8
  integer(c_int64_t), parameter :: I64  = 64_8
  integer(c_int64_t), parameter :: NSEG = SEG / BLK + 1_8
  real(c_double) :: ca, cb, a, boff, segcarry
  real(c_double) :: PVA(64), PVB(64)
  real(c_double) :: BA(NSEG), BB(NSEG), CARR(NSEG)

  n = LEN_1D
  if (n <= 1) return
  call c_f_pointer(pc, c, [n])
  call c_f_pointer(px, x, [n])
  call c_f_pointer(py, y, [n])

  segcarry = y(1)
  segstart = 1
  do while (segstart <= n)
    segend = min(n, segstart + SEG - 1)
    segn   = segend - segstart + 1
    nb     = (segn + BLK - 1) / BLK

    ! ---- P1: endpoint affine map of each block in the segment
    !$omp parallel do schedule(static) default(none) shared(BA, BB, c, x, n, nb, segstart, segend) private(a, boff, lo, hi, s, m, d, j, ch, PVA, PVB)
    do b = 1, nb
      lo = segstart + (b - 1) * BLK
      hi = min(lo + BLK - 1, segend)
      s  = max(lo, I2)
      a = 1.0d0
      boff = 0.0d0
      do ch = s, hi, I64
        m = min(I64, hi - ch + 1)
        do j = 1, m
          PVA(j) = c(ch + j - 1)
          PVB(j) = x(ch + j - 1)
        end do
        d = 1_8
        do while (d < m)
          do j = m, d + 1, -1
            PVB(j) = PVA(j) * PVB(j - d) + PVB(j)
            PVA(j) = PVA(j) * PVA(j - d)
          end do
          d = d * 2
        end do
        a = PVA(m) * a
        boff = PVA(m) * boff + PVB(m)
      end do
      BA(b) = a
      BB(b) = boff
    end do
    !$omp end parallel do

    ! ---- P2: exclusive scan over block maps; CARR(b) = y(lo_b - 1)
    ca = 1.0d0
    cb = 0.0d0
    do b = 1, nb
      CARR(b) = ca * segcarry + cb
      ca = BA(b) * ca
      cb = BA(b) * cb + BB(b)
    end do
    segcarry = ca * segcarry + cb

    ! ---- P3: per-element prefix, apply carry, write y
    !$omp parallel do schedule(static) default(none) shared(CARR, c, x, y, n, nb, segstart, segend) private(a, lo, hi, s, m, d, j, ch, PVA, PVB)
    do b = 1, nb
      lo = segstart + (b - 1) * BLK
      hi = min(lo + BLK - 1, segend)
      s  = max(lo, I2)
      a = CARR(b)
      do ch = s, hi, I64
        m = min(I64, hi - ch + 1)
        do j = 1, m
          PVA(j) = c(ch + j - 1)
          PVB(j) = x(ch + j - 1)
        end do
        d = 1_8
        do while (d < m)
          do j = m, d + 1, -1
            PVB(j) = PVA(j) * PVB(j - d) + PVB(j)
            PVA(j) = PVA(j) * PVA(j - d)
          end do
          d = d * 2
        end do
        do j = 1, m
          y(ch + j - 1) = PVA(j) * a + PVB(j)
        end do
        a = PVA(m) * a + PVB(m)
      end do
    end do
    !$omp end parallel do

    segstart = segend + 1
  end do
end subroutine scan_affine_decay_fp64
