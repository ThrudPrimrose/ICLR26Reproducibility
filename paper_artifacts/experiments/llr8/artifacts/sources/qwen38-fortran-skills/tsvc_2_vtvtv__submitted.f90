subroutine tsvc_2_vtvtv(a, b, c, len_1d, workspace, workspace_size) bind(C, name="tsvc_2_vtvtv_fp64")
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t) :: i

  if (len_1d <= 0) return

  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine tsvc_2_vtvtv
