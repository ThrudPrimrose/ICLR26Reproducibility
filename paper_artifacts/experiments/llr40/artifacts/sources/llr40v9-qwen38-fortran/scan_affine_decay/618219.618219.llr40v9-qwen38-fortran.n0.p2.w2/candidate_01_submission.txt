! Parallel blocked scan for y(i) = c(i)*y(i-1) + x(i),  i = 2..n  (1-based).
! Element 1 is the seed (fixed by the harness); the recurrence is a scan over
! the non-commutative monoid of affine maps f_i(t) = c(i)*t + x(i) for i >= 2.
! Composing (a,b) after (A,B): (A*a, A*b + B).
! Phase 1 (parallel): per-block endpoint map M_b over its elements (>= 2).
! Phase 2 (serial)  : exclusive scan over block maps -> boundary value per block.
! Phase 3 (parallel): per-block chain from its boundary value.
subroutine scan_affine_decay_fp64(pc, px, py, LEN_1D) bind(C, name="scan_affine_decay_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value :: pc, px, py
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), pointer :: c(:), x(:), y(:)

  integer(c_int64_t) :: n, nb, b, i, lo, hi, s
  integer(c_int64_t), parameter :: BLK = 4096_8
  integer(c_int64_t), parameter :: I2 = 2_8
  real(c_double) :: a, boff, seed, ca, cb
  real(c_double), allocatable :: BA(:), BB(:), CARR(:)

  n = LEN_1D
  if (n <= 1) return

  call c_f_pointer(pc, c, [n])
  call c_f_pointer(px, x, [n])
  call c_f_pointer(py, y, [n])

  nb = (n + BLK - 1) / BLK
  allocate(BA(nb), BB(nb), CARR(nb))

  ! Phase 1: endpoint affine map of each block, M_b = (BA(b), BB(b))
  !$omp parallel do schedule(static)
  do b = 1, nb
    a = 1.0d0
    boff = 0.0d0
    lo = (b - 1) * BLK + 1
    hi = min(n, b * BLK)
    s = max(lo, I2)
    do i = s, hi
      a = c(i) * a
      boff = c(i) * boff + x(i)
    end do
    BA(b) = a
    BB(b) = boff
  end do
  !$omp end parallel do

  ! Phase 2: exclusive scan over block maps; CARR(b) = y(lo_b - 1)
  seed = y(1)
  ca = 1.0d0
  cb = 0.0d0
  do b = 1, nb
    CARR(b) = ca * seed + cb
    ca = BA(b) * ca
    cb = BA(b) * cb + BB(b)
  end do

  ! Phase 3: run the chain inside each block from its boundary value
  !$omp parallel do schedule(static)
  do b = 1, nb
    lo = (b - 1) * BLK + 1
    hi = min(n, b * BLK)
    s = max(lo, I2)
    if (lo == 1) then
      a = seed
    else
      a = CARR(b)
    end if
    do i = s, hi
      a = c(i) * a + x(i)
      y(i) = a
    end do
  end do
  !$omp end parallel do

  deallocate(BA, BB, CARR)
end subroutine scan_affine_decay_fp64
