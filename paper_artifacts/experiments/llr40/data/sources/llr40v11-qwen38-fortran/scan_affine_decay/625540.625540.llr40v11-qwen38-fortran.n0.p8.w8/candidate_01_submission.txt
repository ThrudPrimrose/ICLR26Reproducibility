! scan_affine_decay: y[i] = c[i]*y[i-1] + x[i]  (variable-coefficient affine scan)
!
! Parallel strategy: blocked scan over the affine semigroup.
!   Pass 1 (parallel): per-block combined affine map (A_b, B_b):  M_b(v) = A_b*v + B_b
!   Pass 1b (parallel): per-meta-block combined map over its blocks
!   Pass 2 (serial):    carry scan over the meta-blocks (H ~ O(nt) steps)
!   Pass 3 (parallel):  per-meta-block carry to each block start (M ~ O(G/nt) steps, short)
!   Pass 4 (parallel):  recompute the in-block values from the known start
!
! C ABI (canonical order, pointers sorted by name): (c, x, y, LEN_1D)
! Fortran indexing: y(1) = x(1) is the seed (set by the harness); compute y(2..n).

subroutine scan_affine_decay_fp64(c, x, y, LEN_1D) bind(C, name="scan_affine_decay_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in)  :: c(LEN_1D)
  real(c_double), intent(in)  :: x(LEN_1D)
  real(c_double), intent(inout) :: y(LEN_1D)

  integer(c_int64_t) :: n, nt, i
  integer(c_int64_t) :: bs, G, ms, nm
  integer(c_int64_t) :: blk, h, b0, b1, lo, hi
  real(c_double) :: a, bv, vs, ys
  real(c_double), allocatable, save :: blockA(:), blockB(:), bstart(:)
  real(c_double), allocatable, save :: metaA(:), metaB(:), mstart(:)
  integer(c_int64_t), save :: capA = 0, capH = 0

  n = LEN_1D
  if (n <= 1) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1

  ! Small-n fast path: a serial scan beats the parallel overhead.
  if (n - 1 <= 16 * nt) then
    vs = y(1)
    do i = 2, n
      vs = c(i) * vs + x(i)
      y(i) = vs
    end do
    return
  end if

  bs = 128
  G = (n - 1 + bs - 1) / bs            ! number of blocks over y(2..n)
  ms = (G + 4*nt - 1) / (4*nt)         ! blocks per meta-block
  if (ms < 1) ms = 1
  nm = (G + ms - 1) / ms               ! number of meta-blocks

  if (G > capA) then
    if (allocated(blockA)) deallocate(blockA, blockB, bstart)
    allocate(blockA(G), blockB(G), bstart(G))
    capA = G
  end if
  if (nm > capH) then
    if (allocated(metaA)) deallocate(metaA, metaB, mstart)
    allocate(metaA(nm), metaB(nm), mstart(nm))
    capH = nm
  end if

  ! ---- Pass 1: per-block affine maps (parallel over blocks) ----
  !$omp parallel do schedule(static) private(blk, lo, hi, a, bv)
  do blk = 0, G - 1
    lo = 2 + blk * bs
    hi = min(n, 2 + (blk + 1) * bs - 1)
    a = 1.0d0
    bv = 0.0d0
    do i = lo, hi
      bv = c(i) * bv + x(i)
      a  = c(i) * a
    end do
    blockA(blk + 1) = a
    blockB(blk + 1) = bv
  end do
  !$omp end parallel do

  ! ---- Pass 1b: per-meta-block maps (parallel over meta-blocks) ----
  !$omp parallel do schedule(static) private(h, b0, b1, a, bv)
  do h = 0, nm - 1
    b0 = h * ms + 1
    b1 = min(G, (h + 1) * ms)
    a = 1.0d0
    bv = 0.0d0
    do blk = b0, b1
      bv = blockA(blk) * bv + blockB(blk)
      a  = blockA(blk) * a
    end do
    metaA(h + 1) = a
    metaB(h + 1) = bv
  end do
  !$omp end parallel do

  ! ---- Pass 2: serial carry over meta-blocks ----
  ys = y(1)
  mstart(1) = ys
  do h = 1, nm - 1
    ys = metaA(h) * ys + metaB(h)
    mstart(h + 1) = ys
  end do

  ! ---- Pass 3: per-meta-block carry to block starts (parallel over meta-blocks) ----
  !$omp parallel do schedule(static) private(h, b0, b1, vs)
  do h = 0, nm - 1
    b0 = h * ms + 1
    b1 = min(G, (h + 1) * ms)
    vs = mstart(h + 1)
    bstart(b0) = vs
    do blk = b0, b1 - 1
      vs = blockA(blk) * vs + blockB(blk)
      bstart(blk + 1) = vs
    end do
  end do
  !$omp end parallel do

  ! ---- Pass 4: recompute in-block values (parallel over blocks) ----
  !$omp parallel do schedule(static) private(blk, lo, hi, vs)
  do blk = 0, G - 1
    lo = 2 + blk * bs
    hi = min(n, 2 + (blk + 1) * bs - 1)
    vs = bstart(blk + 1)
    do i = lo, hi
      vs = c(i) * vs + x(i)
      y(i) = vs
    end do
  end do
  !$omp end parallel do

end subroutine scan_affine_decay_fp64
