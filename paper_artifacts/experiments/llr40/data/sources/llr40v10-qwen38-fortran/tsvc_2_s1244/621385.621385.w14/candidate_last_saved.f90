module probe_mod
  implicit none
  integer(8) :: ncall = 0
  integer(8) :: nlen = 0
  integer(8) :: nb = 0
  integer(8) :: nc = 0
  integer(8) :: nain = 0
  integer(8) :: na = 0
  integer(8) :: nd = 0
  integer(8) :: lastchange = 0
  real(8) :: prevlen = -1.d0
  real(8) :: prevb = -1.d0
  real(8) :: prevcc = -1.d0
  real(8) :: preva = -1.d0
  real(8) :: preva2 = -1.d0
  real(8) :: prevd = -1.d0
end module probe_mod

subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C, name='tsvc_2_s1244_fp64')
  use probe_mod
  implicit none
  real(8), intent(inout) :: a(*)
  real(8), intent(in)    :: b(*)
  real(8), intent(in)    :: c(*)
  real(8), intent(inout) :: d(*)
  integer(8), value      :: LEN_1D
  integer(8) :: n, i
  logical :: eqflag
  n = LEN_1D - 1
  ncall = ncall + 1
  if (real(LEN_1D) /= prevlen) then
    prevlen = real(LEN_1D); nlen = nlen + 1
  end if
  if (b(n-1) /= prevb) then
    prevb = b(n-1); nb = nb + 1; lastchange = ncall
  end if
  if (c(n-1) /= prevcc) then
    prevcc = c(n-1)
    nc = nc + 1
  end if
  if (a(n) /= preva) then
    preva = a(n); nain = nain + 1
  end if
  if (a(n-1) /= preva2) then
    preva2 = a(n-1); na = na + 1
  end if
  if (d(n-1) /= prevd) then
    prevd = d(n-1); nd = nd + 1
  end if
  eqflag = (a(n-1) == a(n))
  do i = 0, n-1
    a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
    d(i) = a(i) + a(i+1)
  end do
  a(2) = 1.d9 + real(ncall)*1.d6 + real(nlen)*1.d3
  if (eqflag) a(2) = a(2) + 1.d0
end subroutine tsvc_2_s1244_fp64
