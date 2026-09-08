! Blocked parallel scan for the affine decay recurrence
!   y(i) = c(i) * y(i-1) + x(i),  i = 2..n   (y(1) given)
!
! Each step is an affine map v -> c(i)*v + x(i).  Composing a block of steps
! yields (A, B): v -> A*v + B,  with (A,B).(a,b) = (A*a, A*b + B).
!   phase 1: per-block (A, B) maps               [parallel over blocks]
!   phase 2: serial prefix over block maps       [tiny]
!   phase 3: re-run each block with its incoming [parallel over blocks]

! Canonical ABI argument order (abi_contract Sec. 4): pointers sorted by name
! (c, x, y), then scalars (LEN_1D), then the reserved workspace pair (ignored).
subroutine scan_affine_decay_fp64(c, x, y, n) bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  real(c_double), intent(in)    :: c(*)
  real(c_double), intent(in)    :: x(*)
  real(c_double), intent(inout) :: y(*)
  integer(c_int64_t), value     :: n

  integer(c_int64_t) :: nb, b, lo, hi, i, nsteps, blen
  integer            :: nt, nb_i
  real(c_double), allocatable :: ab(:), bb(:), vin(:)
  real(c_double) :: acc, prod

  if (n <= 1) return
  nsteps = n - 1

  nt = max(1, omp_get_max_threads())
  ! number of blocks: a few per thread, but never more than nsteps
  nb = int(8, c_int64_t) * int(nt, c_int64_t)
  if (nb > nsteps) nb = nsteps
  blen = (nsteps + nb - 1) / nb

  nb_i = int(nb)
  allocate(ab(nb_i), bb(nb_i), vin(nb_i))

!$omp parallel do default(none) shared(y, c, x, n, nb, ab, bb, blen) &
!$omp&         private(b, lo, hi, i, acc, prod) schedule(static)
  do b = 0, nb - 1
     lo = 2 + b * blen
     hi = min(lo + blen - 1, n)
     acc = 0d0
     prod = 1d0
     do i = lo, hi
        acc = c(i) * acc + x(i)
        prod = prod * c(i)
     end do
     ab(b + 1) = prod
     bb(b + 1) = acc
  end do
!$omp end parallel do

  acc = y(1)
  do b = 1, nb_i
     vin(b) = acc
     acc = ab(b) * acc + bb(b)
  end do

!$omp parallel do default(none) shared(y, c, x, n, nb, vin, blen) &
!$omp&         private(b, lo, hi, i, acc) schedule(static)
  do b = 0, nb - 1
     lo = 2 + b * blen
     hi = min(lo + blen - 1, n)
     acc = vin(b + 1)
     do i = lo, hi
        acc = c(i) * acc + x(i)
        y(i) = acc
     end do
  end do
!$omp end parallel do

  deallocate(ab, bb, vin)
end subroutine scan_affine_decay_fp64
