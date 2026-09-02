subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D, workspace, workspace_bytes) bind(C, name="tsvc_2_s4112_fp64")
  use iso_c_binding, only: c_int, c_int64_t, c_double, c_ptr
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int), intent(in) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i
  do i = 1, LEN_1D
    a(i) = a(i) + b(ip(i)) * 2.0_c_double
  end do
  ! workspace unused
end subroutine tsvc_2_s4112_fp64
