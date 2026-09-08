subroutine ext_war_unit_fp64(a, b, n) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n)
  integer(c_int64_t) :: i
  do i = 1, n - 1
    a(i) = a(i + 1) + b(i)
  end do
end subroutine ext_war_unit_fp64

subroutine ext_war_unit_alias(a, b, n) bind(C, name="ext_war_unit")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  real(c_double), intent(in) :: b(n)
  interface
    subroutine ext_war_unit_fp64(a, b, n) bind(C)
      use iso_c_binding
      integer(c_int64_t), value, intent(in) :: n
      real(c_double), intent(inout) :: a(n)
      real(c_double), intent(in) :: b(n)
    end subroutine ext_war_unit_fp64
  end interface
  call ext_war_unit_fp64(a, b, n)
end subroutine ext_war_unit_alias
