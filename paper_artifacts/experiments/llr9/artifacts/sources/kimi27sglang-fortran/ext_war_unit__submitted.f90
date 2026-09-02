module ext_war_unit
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  real(c_double), pointer, save :: t(:) => null()
  integer(c_int64_t), save :: last_n = 0
contains
  subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(c)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t), intent(in), value :: LEN_1D
    integer(c_int64_t) :: i, n
    n = LEN_1D - 1
    if (n <= 0) return
    !$omp critical (ext_war_unit_alloc)
    if (n > last_n) then
       if (associated(t)) deallocate(t)
       allocate(t(n))
       last_n = n
    end if
    !$omp end critical (ext_war_unit_alloc)
    !$omp parallel
    !$omp do schedule(static)
    do i = 1, n
       t(i) = a(i + 1)
    end do
    !$omp do schedule(static)
    do i = 1, n
       a(i) = t(i) + b(i)
    end do
    !$omp end parallel
  end subroutine ext_war_unit_fp64
end module ext_war_unit
