subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, s, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)

  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = a(i) + b(i) * s
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vpvts_fp64
