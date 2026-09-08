! TSVC tsvc_2 s252 optimized.
! Reference: t=0; for i: s=b[i]*c[i]; a[i]=s+t; t=s
! Since t(i) == s(i-1): a(1)=b(1)*c(1), a(i)=b(i)*c(i)+b(i-1)*c(i-1).
! No loop-carried dependence -> parallel + vectorizable.
! Both products are stored to temporaries BEFORE the add so the compiler
! cannot contract to FMA: matches the scalar (NumPy) oracle bit-exactly.
subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a
  real(c_double), dimension(*), intent(in)    :: b
  real(c_double), dimension(*), intent(in)    :: c
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i
  real(c_double) :: p0, p1

  if (LEN_1D < 1) return
  a(1) = b(1)*c(1) + 0.0d0
  !$omp parallel do
  do i = 2, LEN_1D
     p1 = b(i)*c(i)
     p0 = b(i-1)*c(i-1)
     a(i) = p1 + p0
  end do
end subroutine tsvc_2_s252_fp64
