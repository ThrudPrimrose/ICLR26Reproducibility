subroutine tsvc_2_s275_fp64(aa, bb, cc, len_2d, ws, wsn) &
     bind(c, name="tsvc_2_s275_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: aa(*)
  real(c_double), intent(in)    :: bb(*)
  real(c_double), intent(in)    :: cc(*)
  integer(c_int64_t), intent(in), value :: len_2d
  integer(c_int64_t), intent(in), value :: ws
  integer(c_int64_t), intent(in), value :: wsn
  integer(c_int64_t) :: m, i, j, nj
  real(c_double) :: r, t1, t2
  m = len_2d
  if (ws /= 0_c_int64_t .or. wsn /= 0_c_int64_t) then
    m = m
  end if
  !$omp parallel do default(none) shared(aa,bb,cc,m) private(i,j,nj,r,t1,t2)
  do i = 0, m-1
    if (aa(1 + i) > 0.0d0) then
      r = aa(1 + i)
      nj = m - 1
      do j = 1, nj - 1, 2
        t1 = bb(1 + j*m + i) * cc(1 + j*m + i)
        t2 = bb(1 + (j+1)*m + i) * cc(1 + (j+1)*m + i)
        r = r + t1
        aa(1 + j*m + i) = r
        r = r + t2
        aa(1 + (j+1)*m + i) = r
      end do
      if (mod(nj, 2_c_int64_t) == 1) then
        r = r + bb(1 + nj*m + i) * cc(1 + nj*m + i)
        aa(1 + nj*m + i) = r
      end if
    end if
  end do
  return
end subroutine tsvc_2_s275_fp64
