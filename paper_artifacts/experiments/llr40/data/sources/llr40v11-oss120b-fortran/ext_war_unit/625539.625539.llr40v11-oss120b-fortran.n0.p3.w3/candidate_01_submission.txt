module ext_war_unit_mod
  use iso_c_binding
  implicit none
contains
  subroutine ext_war_unit_fp64(a, b, len_1d) bind(C, name="ext_war_unit_fp64")
    integer(c_int64_t), value :: len_1d
    real(c_double), intent(inout) :: a(0:len_1d-1)
    real(c_double), intent(in) :: b(0:len_1d-1)
    real(c_double), allocatable :: a_tmp(:)
    integer(c_int64_t) :: i
    if (len_1d <= 1) return
    allocate(a_tmp(0:len_1d-1))
    ! Copy a to a_tmp in parallel
    !$omp parallel do schedule(static)
    do i = 0, len_1d - 1
      a_tmp(i) = a(i)
    end do
    !$omp end parallel do
    ! Compute using a_tmp as source
    !$omp parallel do schedule(static)
    do i = 0, len_1d - 2
      a(i) = a_tmp(i+1) + b(i)
    end do
    !$omp end parallel do
    deallocate(a_tmp)
  end subroutine ext_war_unit_fp64
end module ext_war_unit_mod
