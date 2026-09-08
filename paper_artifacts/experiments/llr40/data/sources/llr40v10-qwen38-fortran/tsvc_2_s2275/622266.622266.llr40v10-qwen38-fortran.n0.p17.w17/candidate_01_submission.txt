subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C, name='tsvc_2_s2275_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value :: a, aa
  type(c_ptr), value :: b, bb, c, cc, d
  integer(c_int64_t), value, intent(in) :: len_2d

  real(c_double), pointer :: pa(:)   => null()
  real(c_double), pointer :: paa(:)  => null()
  real(c_double), pointer :: pb(:)   => null()
  real(c_double), pointer :: pbb(:)  => null()
  real(c_double), pointer :: pc(:)   => null()
  real(c_double), pointer :: pcc(:)  => null()
  real(c_double), pointer :: pd(:)   => null()
  integer(c_int64_t) :: n, n2
  integer(c_int64_t) :: i

  n  = len_2d
  n2 = len_2d * len_2d

  call c_f_pointer(a,  pa,  [n])
  call c_f_pointer(aa, paa, [n2])
  call c_f_pointer(b,  pb,  [n])
  call c_f_pointer(bb, pbb, [n2])
  call c_f_pointer(c,  pc,  [n])
  call c_f_pointer(cc, pcc, [n2])
  call c_f_pointer(d,  pd,  [n])

  !$omp parallel do
  do i = 1, n
    pa(i) = pb(i) + pc(i) * pd(i)
  end do

  !$omp parallel do
  do i = 1, n2
    paa(i) = paa(i) + pbb(i) * pcc(i)
  end do
end subroutine tsvc_2_s2275_fp64
