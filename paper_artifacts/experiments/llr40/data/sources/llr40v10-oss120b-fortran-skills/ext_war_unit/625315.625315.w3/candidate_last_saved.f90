subroutine ext_war_unit(a, b, len_1d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double) :: a_tmp(len_1d)
  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, len_1d
    a_tmp(i) = a(i)
  end do

  !$omp parallel do simd
  do i = 1, len_1d - 1
    a(i) = a_tmp(i + 1) + b(i)
  end do

end subroutine ext_war_unit
