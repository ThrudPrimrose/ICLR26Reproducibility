subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t) :: i
  real(c_double), allocatable :: a_tmp(:)

  if (LEN_1D <= 1_c_int64_t) return

  allocate(a_tmp(LEN_1D))

  !$omp parallel do default(none) shared(a, a_tmp, LEN_1D) private(i) schedule(static)
  do i = 1, LEN_1D
    a_tmp(i) = a(i)
  end do
  !$omp end parallel do

  !$omp parallel do default(none) shared(a, a_tmp, b, LEN_1D) private(i) schedule(static)
  do i = 1, LEN_1D-1
    a(i) = a_tmp(i+1) + b(i)
  end do
  !$omp end parallel do

  deallocate(a_tmp)
end subroutine ext_war_unit_fp64
