subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int32_t), intent(in) :: ip(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = a(i) + b(ip(i)) * 2.0_c_double
  end do
  !$omp end parallel do simd

end subroutine tsvc_2_s4112_fp64
