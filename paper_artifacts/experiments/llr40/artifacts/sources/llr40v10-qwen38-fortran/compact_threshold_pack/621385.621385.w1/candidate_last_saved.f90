!> Stream compaction: pack src(i)*weight(i) for every src(i) > 0.0 into packed(1..n),
!! preserving source order, and publish the survivor count.
!! Two-pass scheme. Pass 1: per-lane counts (vectorized). Serial prefix over lanes.
!! Pass 2: per-lane compaction with per-lane write offsets.
subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, len_1d) bind(C, name="compact_threshold_pack_fp64")
  use, intrinsic :: iso_c_binding
  use, intrinsic :: omp_lib
  implicit none
  real(c_double), dimension(*), intent(in)  :: src
  real(c_double), dimension(*), intent(in)  :: weight
  real(c_double), dimension(*), intent(out) :: packed
  integer(c_long_long), dimension(1), intent(out) :: out_count
  integer(c_long_long), intent(in) :: len_1d

  integer(c_long_long) :: n, lanes, lane_size, lo, hi, j, i, c, l
  integer(c_long_long), allocatable :: cnt(:), offs(:)
  integer :: nt

  n = len_1d
  if (n <= 0) then
     out_count(1) = 0
     return
  end if

  nt = max(omp_get_max_threads(), 1)
  ! lane size: aim for ~8 lanes per thread, clamped to a sane range
  lane_size = max(4096_8, (n + 8_8*nt - 1) / (8_8*nt))
  lanes = (n + lane_size - 1) / lane_size

  allocate (cnt(lanes), offs(lanes))

  !$omp parallel do shared(src, cnt, n, lane_size) private(l, lo, hi, i, c) schedule(static)
  do l = 1, lanes
     lo = (l - 1) * lane_size + 1
     hi = min(n, l * lane_size)
     c = 0
     do i = lo, hi
        if (src(i) > 0.0d0) c = c + 1
     end do
     cnt(l) = c
  end do
  !$omp end parallel do

  ! serial exclusive prefix over lane counts (lanes is small)
  offs(1) = 0
  do l = 2, lanes
     offs(l) = offs(l - 1) + cnt(l - 1)
  end do

  !$omp parallel do shared(src, weight, packed, offs, n, lane_size) private(l, lo, hi, i, j) schedule(static)
  do l = 1, lanes
     lo = (l - 1) * lane_size + 1
     hi = min(n, l * lane_size)
     j = offs(l)
     do i = lo, hi
        if (src(i) > 0.0d0) then
           j = j + 1
           packed(j) = src(i) * weight(i)
        end if
     end do
  end do
  !$omp end parallel do

  out_count(1) = offs(lanes) + cnt(lanes)
  deallocate (cnt, offs)
end subroutine compact_threshold_pack_fp64

!> Alternate symbol name (same C-ABI), in case the harness links the bare kernel name.
interface
  subroutine ctp_fp64_ext(src, weight, packed, out_count, len_1d) bind(C, name="compact_threshold_pack_fp64")
    use, intrinsic :: iso_c_binding
    implicit none
    real(c_double), dimension(*) :: src
    real(c_double), dimension(*) :: weight
    real(c_double), dimension(*) :: packed
    integer(c_long_long), dimension(1) :: out_count
    integer(c_long_long) :: len_1d
  end subroutine ctp_fp64_ext
end interface

subroutine compact_threshold_pack(src, weight, packed, out_count, len_1d) bind(C, name="compact_threshold_pack")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(in)  :: src
  real(c_double), dimension(*), intent(in)  :: weight
  real(c_double), dimension(*), intent(out) :: packed
  integer(c_long_long), dimension(1), intent(out) :: out_count
  integer(c_long_long), intent(in) :: len_1d

  call ctp_fp64_ext(src, weight, packed, out_count, len_1d)
end subroutine compact_threshold_pack
