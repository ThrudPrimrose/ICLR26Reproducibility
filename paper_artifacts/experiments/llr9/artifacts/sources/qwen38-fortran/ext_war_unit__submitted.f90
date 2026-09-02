subroutine ext_war_unit_fp64(a, b, len_1d, workspace, workspace_size) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: n, k, i, lo, hi, nb
  integer :: nt, nch
  real(c_double) :: halo(1024), v(8)

  n = len_1d
  if (n < 2) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  nch = 2 * nt
  if (nch > 1024) nch = 1024
  if (nch > n - 1) nch = int(n - 1)

  !$omp parallel default(none) shared(n, nch, a, b, halo) private(lo, hi, nb, i, v)
  ! Phase 1: each chunk saves the one element a(hi) it needs from the right
  ! neighbour, while every value is still the original (read-only region).
  !$omp do schedule(static)
  do k = 1, nch
    hi = (n - 1) * k / nch + 1
    halo(k) = a(hi)
  end do
  ! Phase 2: ascending order inside a chunk keeps a(i+1) original for all
  ! i < hi-1 (only lower positions are already overwritten). 8-wide blocks
  ! vectorize to 512-bit; v decouples the store from a so no alias versioning.
  !$omp do schedule(static)
  do k = 1, nch
    lo = (n - 1) * (k - 1) / nch + 1
    hi = (n - 1) * k / nch + 1
    nb = (hi - 1 - lo) / 8
    do i = lo, lo + 8 * (nb - 1), 8
      v(1:8) = a(i+1:i+8)
      a(i:i+7) = v(1:8) + b(i:i+7)
    end do
    do i = lo + 8 * nb, hi - 2
      a(i) = a(i + 1) + b(i)
    end do
    a(hi - 1) = halo(k) + b(hi - 1)
  end do
  !$omp end parallel

end subroutine ext_war_unit_fp64
