! TSVC tsvc_2 s3110 - max-with-first-occurrence-index over a 2D array.
! C-ABI entry: void tsvc_2_s3110_fp64(const double* aa, double* bb, int64_t LEN_2D)
subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name='tsvc_2_s3110_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  type(c_ptr), value :: aa, bb
  integer(c_int64_t), value :: LEN_2D

  integer(c_int64_t) :: n, idx, xindex, yindex, lo, hi, gidx
  real(c_double) :: maxv, chksum, lmax
  real(c_double), pointer :: ap(:), bp(:)
  real(c_double), allocatable :: loc_max(:)
  integer(c_int64_t), allocatable :: loc_idx(:)
  integer :: nt, t

  n = LEN_2D * LEN_2D
  call c_f_pointer(aa, ap, [n])
  call c_f_pointer(bb, bp, [4])

  nt = 1
!$ nt = omp_get_max_threads()
  allocate(loc_max(nt), loc_idx(nt))

!$omp parallel default(none) private(t, lo, hi, idx, lmax, maxv) shared(ap, n, nt, loc_max, loc_idx)
!$   t = omp_get_thread_num()
!$   lo = (n * t) / nt
!$   hi = (n * (t + 1)) / nt
!$   maxv = ap(lo + 1)
!$   lmax = maxv
!$   loc_idx(t+1) = lo
!$   do idx = lo, hi - 1
!$      if (ap(idx + 1) > lmax) then
!$         lmax = ap(idx + 1)
!$         loc_idx(t+1) = idx
!$      end if
!$   end do
!$   loc_max(t+1) = lmax
!$omp end parallel

  maxv = loc_max(1); gidx = loc_idx(1)
  do t = 2, nt
     if (loc_max(t) > maxv) then
        maxv = loc_max(t); gidx = loc_idx(t)
     else if (loc_max(t) == maxv .and. loc_idx(t) < gidx) then
        gidx = loc_idx(t)
     end if
  end do

  xindex = gidx / LEN_2D
  yindex = mod(gidx, LEN_2D)
  chksum = maxv + dble(xindex) + dble(yindex)
  bp(1) = chksum
end subroutine tsvc_2_s3110_fp64
