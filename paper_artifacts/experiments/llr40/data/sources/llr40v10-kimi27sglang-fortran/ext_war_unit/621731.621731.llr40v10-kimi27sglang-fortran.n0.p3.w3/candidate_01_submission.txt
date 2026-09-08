subroutine ext_war_unit_fp64(a, b, n, workspace, workspace_size) bind(C, name="ext_war_unit_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value, intent(in) :: n
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  do i = 1, n - 1
    a(i) = a(i + 1) + b(i)
  end do
end subroutine ext_war_unit_fp64
