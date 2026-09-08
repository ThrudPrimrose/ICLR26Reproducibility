subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, len_1d, workspace, workspace_size) bind(C, name="compact_threshold_pack_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  integer(c_int64_t), intent(inout) :: out_count(*)
  real(c_double), intent(inout) :: packed(*)
  real(c_double), intent(in) :: src(*)
  real(c_double), intent(in) :: weight(*)
  integer(c_int64_t), value, intent(in) :: len_1d
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i, b, n, nb, lo, hi, c, run
  integer(c_int64_t), allocatable :: cnt(:), off(:)

  n = len_1d
  if (n <= 0) then
    out_count(1) = 0_8
    return
  end if
  nb = (n + 16383_8) / 16384_8
  allocate(cnt(nb), off(nb))

  ! stage 1: per-block survivor counts (vectorizable inner loop)
  !$omp parallel do
  do b = 1, nb
    lo = (b - 1_8) * 16384_8 + 1_8
    hi = min(b * 16384_8, n)
    c = 0_8
    do i = lo, hi
      c = c + merge(1_8, 0_8, src(i) > 0.0d0)
    end do
    cnt(b) = c
  end do

  ! stage 2: prefix over block counts (small)
  run = 0_8
  do b = 1, nb
    off(b) = run
    run = run + cnt(b)
  end do

  ! stage 3: in-block scan + scatter
  !$omp parallel do
  do b = 1, nb
    lo = (b - 1_8) * 16384_8 + 1_8
    hi = min(b * 16384_8, n)
    c = off(b)
    do i = lo, hi
      if (src(i) > 0.0d0) then
        c = c + 1_8
        packed(c) = src(i) * weight(i)
      end if
    end do
  end do

  out_count(1) = run
  deallocate(cnt, off)
end subroutine compact_threshold_pack_fp64
