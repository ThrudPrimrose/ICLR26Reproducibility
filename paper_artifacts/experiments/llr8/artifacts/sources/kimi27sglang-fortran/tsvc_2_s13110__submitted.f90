subroutine tsvc_2_s13110_fp64(aa, bb, LEN_2D) bind(C, name='tsvc_2_s13110_fp64')
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value :: LEN_2D
  real(c_double), intent(in) :: aa(LEN_2D*LEN_2D)
  real(c_double), intent(out) :: bb(2, 2)

  integer(c_int64_t) :: n, nt, tid, k, istart, iend, loc
  real(c_double) :: lmaxv, v
  integer(c_int64_t) :: lxi_loc, lyi_loc
  real(c_double) :: maxv, chksum
  integer(c_int64_t) :: xindex, yindex
  real(c_double), allocatable :: lmax(:)
  integer(c_int64_t), allocatable :: lxi(:), lyi(:)

  n = LEN_2D * LEN_2D
  nt = omp_get_max_threads()
  allocate(lmax(nt), lxi(nt), lyi(nt))

  !$omp parallel default(none) shared(aa, LEN_2D, n, nt, lmax, lxi, lyi) private(tid, k, v, loc, istart, iend, lmaxv, lxi_loc, lyi_loc)
  tid = omp_get_thread_num() + 1
  istart = (n * (tid - 1_c_int64_t)) / nt + 1_c_int64_t
  iend = (n * tid) / nt
  lmaxv = aa(istart)
  loc = istart - 1_c_int64_t
  lxi_loc = loc / LEN_2D
  lyi_loc = mod(loc, LEN_2D)
  do k = istart, iend
    v = aa(k)
    if (v > lmaxv) then
      lmaxv = v
      loc = k - 1_c_int64_t
      lxi_loc = loc / LEN_2D
      lyi_loc = mod(loc, LEN_2D)
    end if
  end do
  lmax(tid) = lmaxv
  lxi(tid) = lxi_loc
  lyi(tid) = lyi_loc
  !$omp end parallel

  maxv = lmax(1)
  xindex = lxi(1)
  yindex = lyi(1)
  do k = 2, nt
    if (lmax(k) > maxv .or. (lmax(k) == maxv .and. (lxi(k) < xindex .or. (lxi(k) == xindex .and. lyi(k) < yindex)))) then
      maxv = lmax(k)
      xindex = lxi(k)
      yindex = lyi(k)
    end if
  end do

  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1, 1) = chksum
end subroutine tsvc_2_s13110_fp64
