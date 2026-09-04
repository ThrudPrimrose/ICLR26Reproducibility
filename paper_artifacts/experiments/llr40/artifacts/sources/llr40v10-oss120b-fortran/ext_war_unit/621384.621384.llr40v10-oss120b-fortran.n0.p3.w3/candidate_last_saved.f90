module ext_war_unit_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C, name='ext_war_unit_fp64')
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t) :: i
    real(c_double), allocatable :: a_tmp(:)

    allocate(a_tmp(LEN_1D))

    !$omp parallel default(none) shared(a,b,a_tmp,LEN_1D) private(i)
    !$omp do schedule(static)
    do i = 1, LEN_1D
      a_tmp(i) = a(i)
    end do
    !$omp end do
    !$omp do schedule(static)
    do i = 1, LEN_1D-1
      a(i) = a_tmp(i+1) + b(i)
    end do
    !$omp end do
    !$omp end parallel

    deallocate(a_tmp)
  end subroutine ext_war_unit_fp64
end module ext_war_unit_mod