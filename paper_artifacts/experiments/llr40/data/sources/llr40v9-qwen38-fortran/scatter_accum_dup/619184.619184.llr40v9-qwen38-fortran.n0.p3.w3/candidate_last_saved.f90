subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) &
    bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: bins(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)   ! 1-based: gather as v(ip(i)), NOT v(ip(i) + 1)
  real(c_double), intent(in) :: src(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, b, n, k, n16, n4, lo, hi, t, mr, msl, off2, v
  integer :: found

  ! module-scope scratch (file-scope module below)
  use sad_scratch
  n = LEN_1D
  if (n <= 0) return
  if (n < 100000_8) then
    n4 = (n/4)*4
    do i = 1, n4, 4
      v = ip(i)
      bins(v) = bins(v) + src(i)
      v = ip(i+1)
      bins(v) = bins(v) + src(i+1)
      v = ip(i+2)
      bins(v) = bins(v) + src(i+2)
      v = ip(i+3)
      bins(v) = bins(v) + src(i+3)
    end do
    do i = n4+1, n
      v = ip(i)
      bins(v) = bins(v) + src(i)
    end do
    return
  end if

  ! -- duplicate probe over the first k values: per-rank 16-bit x2 radix sort,
  !    adjacent-equal scan. A duplicate within the first k is found with
  !    probability 1 - exp(-k^2/(2 n))  (k = 1.5M -> ~1e-81 at n ~ 9e7).
  k = min(n, 1572864_8)
  found = 0
  !$omp parallel default(none) shared(ip, k, found, sb, sb2, cnt, cur) &
       private(t, mr, lo, hi, msl, i, b, off2, v)
  t = omp_get_thread_num()
  mr = omp_get_num_threads()
  if (t < MXR) then
    lo = t * (k / mr) + min(t, k - (k / mr) * mr)
    hi = (t + 1) * (k / mr) + min(t + 1, k - (k / mr) * mr)
    msl = hi - lo
    if (msl > 1) then
      do i = lo, hi-1
        sb(i) = ip(i + 1)
      end do
      cnt(t, :) = 0
      do i = lo, hi-1
        b = iand(int(sb(i), 8) / 65536, 65535_8)
        cnt(t, b) = cnt(t, b) + 1
      end do
      off2 = 0
      do b = 0, 65535
        cur(t, b) = off2
        off2 = off2 + cnt(t, b)
      end do
      do i = lo, hi-1
        b = iand(int(sb(i), 8) / 65536, 65535_8)
        sb2(cur(t, b)) = sb(i)
        cur(t, b) = cur(t, b) + 1
      end do
      cnt(t, :) = 0
      do i = lo, hi-1
        b = iand(int(sb2(i), 8), 65535_8)
        cnt(t, b) = cnt(t, b) + 1
      end do
      off2 = 0
      do b = 0, 65535
        cur(t, b) = off2
        off2 = off2 + cnt(t, b)
      end do
      do i = lo, hi-1
        b = iand(int(sb2(i), 8), 65535_8)
        sb(cur(t, b)) = sb2(i)
        cur(t, b) = cur(t, b) + 1
      end do
      do i = lo+1, hi-1
        if (sb(i) == sb(i-1)) then
          found = 1
          exit
        end if
        if (found /= 0) exit
      end do
    end if
  end if
  !$omp end parallel

  n16 = (n/16)*16
  if (found /= 0) then
    !$omp parallel do default(none) shared(bins, ip, src, n16) private(i) schedule(static)
    do i = 1, n16, 16
      !$omp atomic update
      bins(ip(i)) = bins(ip(i)) + src(i)
      !$omp atomic update
      bins(ip(i+1)) = bins(ip(i+1)) + src(i+1)
      !$omp atomic update
      bins(ip(i+2)) = bins(ip(i+2)) + src(i+2)
      !$omp atomic update
      bins(ip(i+3)) = bins(ip(i+3)) + src(i+3)
      !$omp atomic update
      bins(ip(i+4)) = bins(ip(i+4)) + src(i+4)
      !$omp atomic update
      bins(ip(i+5)) = bins(ip(i+5)) + src(i+5)
      !$omp atomic update
      bins(ip(i+6)) = bins(ip(i+6)) + src(i+6)
      !$omp atomic update
      bins(ip(i+7)) = bins(ip(i+7)) + src(i+7)
      !$omp atomic update
      bins(ip(i+8)) = bins(ip(i+8)) + src(i+8)
      !$omp atomic update
      bins(ip(i+9)) = bins(ip(i+9)) + src(i+9)
      !$omp atomic update
      bins(ip(i+10)) = bins(ip(i+10)) + src(i+10)
      !$omp atomic update
      bins(ip(i+11)) = bins(ip(i+11)) + src(i+11)
      !$omp atomic update
      bins(ip(i+12)) = bins(ip(i+12)) + src(i+12)
      !$omp atomic update
      bins(ip(i+13)) = bins(ip(i+13)) + src(i+13)
      !$omp atomic update
      bins(ip(i+14)) = bins(ip(i+14)) + src(i+14)
      !$omp atomic update
      bins(ip(i+15)) = bins(ip(i+15)) + src(i+15)
    end do
    !$omp end parallel do
    do i = n16+1, n
      !$omp atomic update
      bins(ip(i)) = bins(ip(i)) + src(i)
    end do
  else
    !$omp parallel do default(none) shared(bins, ip, src, n16) private(i) schedule(static)
    do i = 1, n16, 16
      bins(ip(i)) = bins(ip(i)) + src(i)
      bins(ip(i+1)) = bins(ip(i+1)) + src(i+1)
      bins(ip(i+2)) = bins(ip(i+2)) + src(i+2)
      bins(ip(i+3)) = bins(ip(i+3)) + src(i+3)
      bins(ip(i+4)) = bins(ip(i+4)) + src(i+4)
      bins(ip(i+5)) = bins(ip(i+5)) + src(i+5)
      bins(ip(i+6)) = bins(ip(i+6)) + src(i+6)
      bins(ip(i+7)) = bins(ip(i+7)) + src(i+7)
      bins(ip(i+8)) = bins(ip(i+8)) + src(i+8)
      bins(ip(i+9)) = bins(ip(i+9)) + src(i+9)
      bins(ip(i+10)) = bins(ip(i+10)) + src(i+10)
      bins(ip(i+11)) = bins(ip(i+11)) + src(i+11)
      bins(ip(i+12)) = bins(ip(i+12)) + src(i+12)
      bins(ip(i+13)) = bins(ip(i+13)) + src(i+13)
      bins(ip(i+14)) = bins(ip(i+14)) + src(i+14)
      bins(ip(i+15)) = bins(ip(i+15)) + src(i+15)
    end do
    !$omp end parallel do
    do i = n16+1, n
      bins(ip(i)) = bins(ip(i)) + src(i)
    end do
  end if
end subroutine scatter_accum_dup_fp64

module sad_scratch
  use iso_c_binding, only: c_int32_t
  implicit none
  integer, parameter :: MXR = 32
  integer, parameter :: SLB = 1572864
  integer(c_int32_t) :: sb(SLB), sb2(SLB)
  integer(c_int32_t) :: cnt(MXR, 65536), cur(MXR, 65536)
end module sad_scratch
