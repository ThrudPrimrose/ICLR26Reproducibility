! Stream compaction: pack src(i)*weight(i) for every src(i) > 0, publish the count.
! Canonical C-ABI (outputs first): out_count, packed, src, weight, LEN_1D, workspace, workspace_size
subroutine compact_threshold_pack_fp64(oc, p, src, w, n, workspace, workspace_size) &
    bind(C, name="compact_threshold_pack_fp64")
  use iso_c_binding
  use omp_lib, only: omp_get_max_threads
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t), intent(inout) :: oc(1)
  real(c_double), intent(inout) :: p(n)
  real(c_double), intent(in) :: src(n)
  real(c_double), intent(in) :: w(n)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: bcnt(1024), off(1024)
  integer(c_int64_t) :: nb, nt, b, i, k, lo, hi, total

  if (n <= 0) return

  nt = omp_get_max_threads()
  if (nt < 1) nt = 1
  nb = nt * 8
  if (nb > 1024) nb = 1024
  if (nb > n) nb = n

  ! pass 1: per-block survivor counts
  !$omp parallel do schedule(static)
  do b = 1, nb
    k = 0
    lo = 1 + (b - 1) * (n / nb)
    hi = lo + (n / nb) - 1
    if (b == nb) hi = n
    do i = lo, hi
      if (src(i) > 0.0d0) k = k + 1
    end do
    bcnt(b) = k
  end do

  ! exclusive prefix over block counts
  off(1) = 1
  do b = 2, nb
    off(b) = off(b - 1) + bcnt(b - 1)
  end do
  total = off(nb) + bcnt(nb) - 1

  ! pass 2: scatter survivors in source order
  !$omp parallel do schedule(static)
  do b = 1, nb
    k = off(b)
    lo = 1 + (b - 1) * (n / nb)
    hi = lo + (n / nb) - 1
    if (b == nb) hi = n
    do i = lo, hi
      if (src(i) > 0.0d0) then
        p(k) = src(i) * w(i)
        k = k + 1
      end if
    end do
  end do

  oc(1) = total
end subroutine compact_threshold_pack_fp64
