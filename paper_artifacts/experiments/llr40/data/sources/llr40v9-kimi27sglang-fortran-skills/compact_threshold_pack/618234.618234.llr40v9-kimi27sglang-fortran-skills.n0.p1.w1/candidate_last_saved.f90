subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, n, workspace, workspace_size) bind(C, name="compact_threshold_pack_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_int8_t, c_ptr, c_f_pointer, c_loc
  implicit none
  integer(c_int64_t), value, intent(in) :: n, workspace_size
  integer(c_int64_t), intent(out) :: out_count(1)
  real(c_double), intent(out) :: packed(n)
  real(c_double), intent(in) :: src(n), weight(n)
  integer(c_int8_t), target, intent(inout) :: workspace(workspace_size)

  integer(c_int64_t), parameter :: BS = 16384
  integer(c_int64_t), parameter :: WBITS = 64
  integer :: b, j
  integer(c_int64_t) :: nblocks, lo, hi, c, total
  integer(c_int64_t) :: nw, wstart, idx, w, k, kmax, wi, nb, it
  integer(c_int64_t), allocatable :: cnt(:), off(:)
  integer(c_int64_t), pointer :: bits(:)
  integer(c_int64_t) :: byte_pos(0:255, 0:7), byte_cnt(0:255)
  type(c_ptr) :: wsp
  logical :: use_workspace
  integer :: bval, p

  nw = (n + WBITS - 1) / WBITS
  nblocks = (n + BS - 1) / BS
  allocate(cnt(0:nblocks-1), off(0:nblocks))
  cnt = 0
  off = 0

  use_workspace = (workspace_size >= nw * 8)
  if (use_workspace) then
    wsp = c_loc(workspace(1))
    call c_f_pointer(wsp, bits, [nw])
  else
    allocate(bits(1:nw))
  end if
  bits = 0

  do bval = 0, 255
    byte_cnt(bval) = int(popcnt(bval), c_int64_t)
    p = 0
    do k = 0, 7
      if (btest(bval, int(k))) then
        byte_pos(bval, p) = k
        p = p + 1
      end if
    end do
    do k = p, 7
      byte_pos(bval, p) = 0
    end do
  end do

  !$omp parallel private(b, lo, hi, wstart, wi, w, k, kmax, j, idx, c, nb, it, bval, p) shared(cnt, off, bits, total, byte_pos, byte_cnt)

  !$omp do schedule(dynamic, 1)
  do b = 0, int(nblocks - 1)
    lo = int(b, c_int64_t) * BS + 1
    hi = min(n, lo + BS - 1)
    wstart = (lo - 1) / WBITS
    nb = (hi - lo + 1 + WBITS - 1) / WBITS
    c = 0
    do wi = 0, nb - 1
      w = 0
      idx = (wstart + wi) * WBITS
      kmax = min(WBITS - 1, hi - idx - 1)
      !$omp simd reduction(ior:w)
      do j = 0, int(kmax)
        if (src(idx + int(j, c_int64_t) + 1) > 0.0d0) w = ibset(w, j)
      end do
      bits(wstart + wi + 1) = w
      c = c + int(popcnt(w), c_int64_t)
    end do
    cnt(b) = c
  end do
  !$omp end do

  !$omp barrier

  !$omp single
  total = 0
  do it = 0, nblocks - 1
    off(it) = total
    total = total + cnt(it)
  end do
  off(nblocks) = total
  !$omp end single

  !$omp barrier

  !$omp do schedule(dynamic, 1)
  do b = 0, int(nblocks - 1)
    lo = int(b, c_int64_t) * BS + 1
    hi = min(n, lo + BS - 1)
    wstart = (lo - 1) / WBITS
    nb = (hi - lo + 1 + WBITS - 1) / WBITS
    c = off(b)
    do wi = 0, nb - 1
      w = bits(wstart + wi + 1)
      do bval = 0, 7
        k = int(iand(ishft(w, -bval*8), int(255, c_int64_t)), c_int64_t)
        do p = 0, int(byte_cnt(k)) - 1
          idx = (wstart + wi) * WBITS + int(bval*8, c_int64_t) + byte_pos(k, p) + 1
          c = c + 1
          packed(c) = src(idx) * weight(idx)
        end do
      end do
    end do
  end do
  !$omp end do

  !$omp end parallel

  out_count(1) = total
  deallocate(cnt, off)
  if (.not. use_workspace) deallocate(bits)
end subroutine compact_threshold_pack_fp64
