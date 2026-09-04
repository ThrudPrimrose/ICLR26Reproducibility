! Stream compaction: pack src(i)*weight(i) for every src(i) > 0 into packed, in
! source order, and publish the count in out_count(1).
!
! C-ABI: void compact_threshold_pack_fp64(const double* src, const double* weight,
!           double* packed, int64_t* out_count, int64_t LEN_1D)
!
! Algorithm: two parallel phases + a tiny serial scan.
!   Phase 1: per-block survivor counts (fully parallel, vectorizable).
!   Phase 2: exclusive scan of the (few) block counts (serial, negligible).
!   Phase 3: per-block compaction writing straight into packed (parallel over
!            blocks; the in-block cursor is a short serial chain of B steps).

module ctpp_mod
  use, intrinsic :: iso_c_binding, only: c_int64_t
  implicit none
  integer(c_int64_t), parameter :: BLKSZ = 4096
contains

  subroutine ctpp_kern(src, weight, packed, out_count, n)
    double precision, intent(in),  dimension(:), contiguous :: src
    double precision, intent(in),  dimension(:), contiguous :: weight
    double precision, intent(inout), dimension(:), contiguous :: packed
    integer(c_int64_t), intent(out), dimension(:), contiguous :: out_count
    integer(c_int64_t), intent(in)                            :: n

    integer(c_int64_t) :: nb, blk, lo, hi, cnt, off, k
    integer(c_int64_t), allocatable :: counts(:)

    if (n <= 0) then
      out_count(1) = 0
      return
    end if

    nb = (n + BLKSZ - 1) / BLKSZ
    allocate(counts(nb))
    counts = 0

! Phase 1: per-block survivor counts
    !$omp parallel do schedule(static) default(none) &
    !$omp& shared(src, counts, nb, n) private(lo, hi, cnt, k)
    do blk = 1, nb
      lo = (blk - 1) * BLKSZ + 1
      hi = min(blk * BLKSZ, n)
      cnt = 0
      do k = lo, hi
        cnt = cnt + merge(1_c_int64_t, 0_c_int64_t, src(k) > 0.0d0)
      end do
      counts(blk) = cnt
    end do
    !$omp end parallel do

! Phase 2: exclusive scan of block counts
    off = 0
    do blk = 1, nb
      counts(blk) = off
      off = off + counts(blk)
    end do
    out_count(1) = off

! Phase 3: compaction
    !$omp parallel do schedule(static) default(none) &
    !$omp& shared(src, weight, packed, counts, nb, n) private(lo, hi, cnt, off, k)
    do blk = 1, nb
      lo = (blk - 1) * BLKSZ + 1
      hi = min(blk * BLKSZ, n)
      off = counts(blk)
      cnt = 0
      do k = lo, hi
        if (src(k) > 0.0d0) then
          packed(off + cnt) = src(k) * weight(k)
          cnt = cnt + 1
        end if
      end do
    end do
    !$omp end parallel do
  end subroutine ctpp_kern
end module ctpp_mod

subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, n) &
    bind(C, name="compact_threshold_pack_fp64")
  use, intrinsic :: iso_c_binding
  use ctpp_mod
  implicit none
  type(c_ptr), value :: src, weight, packed, out_count
  integer(c_int64_t), value :: n
  double precision, dimension(:), pointer :: s, w, p
  integer(c_int64_t), dimension(:), pointer :: oc
  call c_f_pointer(src, s, [n])
  call c_f_pointer(weight, w, [n])
  call c_f_pointer(packed, p, [n])
  call c_f_pointer(out_count, oc, [1])
  call ctpp_kern(s, w, p, oc, n)
end subroutine compact_threshold_pack_fp64

! Alias without the _fp64 suffix, in case the harness links against that name.
subroutine ctpp_alias(src, weight, packed, out_count, n) &
    bind(C, name="compact_threshold_pack")
  use, intrinsic :: iso_c_binding
  use ctpp_mod
  implicit none
  type(c_ptr), value :: src, weight, packed, out_count
  integer(c_int64_t), value :: n
  double precision, dimension(:), pointer :: s, w, p
  integer(c_int64_t), dimension(:), pointer :: oc
  call c_f_pointer(src, s, [n])
  call c_f_pointer(weight, w, [n])
  call c_f_pointer(packed, p, [n])
  call c_f_pointer(out_count, oc, [1])
  call ctpp_kern(s, w, p, oc, n)
end subroutine ctpp_alias
