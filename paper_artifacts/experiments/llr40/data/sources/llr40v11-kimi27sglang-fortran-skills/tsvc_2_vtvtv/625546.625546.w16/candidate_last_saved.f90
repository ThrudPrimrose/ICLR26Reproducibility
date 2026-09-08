subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i
  !$omp parallel do simd simdlen(8)
  do i = 1, len_1d
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine tsvc_2_vtvtv_fp64
