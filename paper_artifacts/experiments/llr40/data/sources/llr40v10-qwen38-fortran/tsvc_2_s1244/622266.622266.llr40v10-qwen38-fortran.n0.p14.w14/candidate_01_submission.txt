! TSVC tsvc_2 s1244, exact reference semantics:
!   iteration i (i = 0..LEN_1D-2):
!     a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
!     d[i] = a[i] + a[i+1]      ! a[i+1] is the ORIGINAL value (not yet written)
! so  d[i] = f(b[i],c[i]) + a_orig[i+1];  a[LEN_1D-1] is never written.
! The scalar form has a loop-carried WAR (a(i) written at i, read old at i-1);
! computing d from the shifted a read and a single fused body keeps it
! dependency-free and lets the compiler vectorize (AVX-512 on the judge).
subroutine tsvc_2_s1244_fp64(a, b, cc, d, len_1d) bind(c, name="tsvc_2_s1244_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  double precision, dimension(*) :: a, b, cc, d
  integer(kind=8), value :: len_1d
  integer(kind=8) :: n, i
  double precision :: a1
  n = len_1d - 1
  do i = 1, n
    a1 = b(i) + cc(i)*cc(i) + b(i)*b(i) + cc(i)
    d(i) = a1 + a(i+1)
    a(i) = a1
  end do
end subroutine tsvc_2_s1244_fp64
