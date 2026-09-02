subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*), c(*)
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = a(i) * b(i) * c(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vtvtv_fp64
