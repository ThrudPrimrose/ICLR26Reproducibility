module ext_war_unit_mod
  use, intrinsic :: iso_c_binding
  implicit none
  private
  ! Cached scratch so we do not re-mmap GBs on every timed call.
  real(c_double), dimension(:), allocatable :: g_tmp
  integer(c_int64_t) :: g_tmp_len = 0
contains

  subroutine ext_war_unit_fp64(a, b, LEN_1D) bind(C, name="ext_war_unit_fp64")
    real(c_double), dimension(*), intent(inout) :: a
    real(c_double), dimension(*), intent(in)    :: b
    integer(c_int64_t), value, intent(in)       :: LEN_1D
    integer(c_int64_t) :: n, j

    if (LEN_1D <= 1) return
    n = LEN_1D

    if (n > g_tmp_len) then
      if (allocated(g_tmp)) deallocate(g_tmp)
      allocate(g_tmp(n))
      g_tmp_len = n
    end if

    ! Snapshot the shifted source: tmp(j) = a(j+1), j = 1..n-1  (Fortran 1-based)
    !$omp parallel do schedule(static)
    do j = 1, n-1
      g_tmp(j) = a(j+1)
    end do

    ! a(j) = original a(j+1) + b(j) = tmp(j) + b(j), j = 1..n-1 ; a(n) untouched
    !$omp parallel do schedule(static)
    do j = 1, n-1
      a(j) = g_tmp(j) + b(j)
    end do
  end subroutine ext_war_unit_fp64
end module ext_war_unit_mod
