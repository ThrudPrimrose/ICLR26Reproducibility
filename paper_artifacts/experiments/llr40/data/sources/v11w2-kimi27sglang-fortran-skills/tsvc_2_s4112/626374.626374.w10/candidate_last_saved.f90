subroutine tsvc_2_s4112_fp64(a, b, ip, n, workspace, workspace_size) bind(C, name="tsvc_2_s4112_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int32_t), intent(in) :: ip(*)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer :: i

  !$omp parallel do simd schedule(static)
  do i = 1, int(n)
    a(i) = a(i) + b(ip(i)) * 2.0_c_double
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s4112_fp64
