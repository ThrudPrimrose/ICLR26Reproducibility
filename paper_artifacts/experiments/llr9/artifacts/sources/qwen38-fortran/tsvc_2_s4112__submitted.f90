subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d, workspace, workspace_size) bind(C, name="tsvc_2_s4112_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t) :: i

!$omp parallel do schedule(static)
  do i = 1, len_1d
    a(i) = a(i) + 2.0d0 * b(ip(i))
  end do
end subroutine tsvc_2_s4112_fp64
