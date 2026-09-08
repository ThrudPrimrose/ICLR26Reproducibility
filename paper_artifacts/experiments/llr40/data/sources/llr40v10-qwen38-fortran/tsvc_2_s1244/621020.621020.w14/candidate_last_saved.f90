! TSVC tsvc_2 s1244 optimized:
!   a[i] = b[i] + c[i]^2 + b[i]^2 + c[i]
!   d[i] = a[i](new) + a[i+1](original -- not yet written when read in scalar loop)
! Fission into two independent elementwise loops (d first, then a) so each is
! fully parallel + SIMD without races, at the minimum 40 B/element traffic.
subroutine tsvc_2_s1244_fp64(pa, pb, pc, pd, len1d) bind(C, name="tsvc_2_s1244_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
  type(c_ptr), intent(in) :: pa, pb, pc, pd
  integer(c_int64_t), intent(in) :: len1d
  real(c_double), dimension(:), pointer :: ash => null(), bsh => null(), &
    csh => null(), dsh => null()
  real(c_double), pointer :: a => null(), b => null(), c => null(), d => null()
  integer(c_int64_t) :: n, i

  n = len1d
  if (n < 2) return
  call c_f_pointer(pa, ash, [n])
  call c_f_pointer(pb, bsh, [n])
  call c_f_pointer(pc, csh, [n])
  call c_f_pointer(pd, dsh, [n])
  a => ash(0:n-1)
  b => bsh(0:n-1)
  c => csh(0:n-1)
  d => dsh(0:n-1)

  !$omp parallel default(none) shared(a,b,c,d,n)
  !$omp do simd default(none) shared(a,b,c,d,n)
  do i = 0, n-2
    d(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i) + a(i+1)
  end do
  !$omp do simd default(none) shared(a,b,c,d,n)
  do i = 0, n-2
    a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
  end do
  !$omp end parallel
end subroutine tsvc_2_s1244_fp64
