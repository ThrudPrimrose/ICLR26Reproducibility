subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(c)
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), allocatable :: t(:)
  integer(c_int64_t) :: i, n

  n = LEN_1D - 1
  if (n <= 0) return
  allocate(t(n))
  !$omp parallel do default(none) shared(a,b,t,n) schedule(static) if(n>1024)
  do i = 1, n
    t(i) = a(i + 1)
  end do
  !$omp parallel do default(none) shared(a,b,t,n) schedule(static) if(n>1024)
  do i = 1, n
    a(i) = t(i) + b(i)
  end do
  deallocate(t)
end subroutine ext_war_unit_fp64
