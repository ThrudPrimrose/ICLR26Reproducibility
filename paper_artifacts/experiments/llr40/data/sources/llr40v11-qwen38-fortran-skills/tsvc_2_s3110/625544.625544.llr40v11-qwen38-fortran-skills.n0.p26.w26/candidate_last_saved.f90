subroutine tsvc_2_s3110_fp64(aa, bb, len_2d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(in) :: aa(len_2d * len_2d)
  real(c_double), intent(inout) :: bb(2, 2)
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: nn, k, first, xindex, yindex
  real(c_double) :: maxv, chksum
  ! per-thread chunk maxima (chunk-filtering pass), persistent scratch
  real(c_double), save :: mpart(0:4095)

  integer(c_int64_t) :: nt, t, lo, hi, lo_w, hi_w, sat, w
  real(c_double) :: mv, gmax

  nn = len_2d * len_2d
  nt = omp_get_max_threads()

  if (nt >= 2 .and. nn >= 64000000 .and. nt <= 4096) then
    ! ---- fast path: one full read (per-thread chunk maxima) + one read of
    ! ---- only the chunk that contains the global maximum (N/nt bytes)
    !$omp parallel
    t = omp_get_thread_num()
    lo = (nn * t) / nt + 1
    hi = (nn * (t + 1)) / nt
    mv = -huge(0.0d0)
    !$omp simd reduction(max:mv)
    do k = lo, hi
      mv = max(mv, aa(k))
    end do
    mpart(t) = mv
    !$omp barrier
    !$omp single
    gmax = -huge(0.0d0)
    w = 0
    do t = 0, nt - 1
      if (mpart(t) > gmax) then
        gmax = mpart(t)
        w = t
      end if
    end do
    lo_w = (nn * w) / nt + 1
    hi_w = (nn * (w + 1)) / nt
    !$omp end single
    !$omp barrier
    sat = nn
    first = sat
    !$omp do simd reduction(min:first)
    do k = lo_w, hi_w
      first = min(first, merge(k - 1, sat, aa(k) == gmax))
    end do
    !$omp single
    xindex = first / len_2d
    yindex = mod(first, len_2d)
    chksum = gmax + dble(xindex)
    chksum = chksum + dble(yindex)
    bb(1, 1) = chksum
    !$omp end single
    !$omp end parallel
  else
    ! ---- serial (or tiny) path: two plain vectorizable passes
    maxv = -huge(0.0d0)
    do k = 1, nn
      maxv = max(maxv, aa(k))
    end do
    first = nn
    do k = 1, nn
      first = min(first, merge(k - 1, nn, aa(k) == maxv))
    end do
    xindex = first / len_2d
    yindex = mod(first, len_2d)
    chksum = maxv + dble(xindex)
    chksum = chksum + dble(yindex)
    bb(1, 1) = chksum
  end if
end subroutine tsvc_2_s3110_fp64
