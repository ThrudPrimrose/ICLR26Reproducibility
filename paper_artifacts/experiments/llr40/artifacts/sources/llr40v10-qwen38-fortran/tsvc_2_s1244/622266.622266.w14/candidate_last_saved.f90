! TSVC tsvc_2 s1244 -- split the fused loop so both phases vectorize:
!   phase 1: a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]   (i = 0..LEN_1D-2)
!   phase 2: d[i] = a[i] + a[i+1]                          (i = 0..LEN_1D-2)
! The original loop is unvectorizable: d[i] reads a[i+1] written one
! iteration later. a[LEN_1D-1] keeps its original value (never written).
subroutine tsvc_2_s1244_fp64(a, b, cc, d, len_1d) bind(c, name="tsvc_2_s1244_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  double precision, dimension(*) :: a, b, cc, d
  integer(kind=8), value :: len_1d
  integer(kind=8) :: n, i

  n = len_1d - 1
  do i = 1, n
    a(i) = b(i) + cc(i)*cc(i) + b(i)*b(i) + cc(i)
  end do
  do i = 1, n
    d(i) = a(i) + a(i + 1)
  end do
end subroutine tsvc_2_s1244_fp64
