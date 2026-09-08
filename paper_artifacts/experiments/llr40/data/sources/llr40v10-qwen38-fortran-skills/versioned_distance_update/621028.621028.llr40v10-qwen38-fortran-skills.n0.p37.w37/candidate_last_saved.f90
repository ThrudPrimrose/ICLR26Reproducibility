subroutine versioned_distance_update_fp64(a, b, c, k, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, k, workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t) :: i
  do i = k + 1, len_1d
    a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
  end do
end subroutine versioned_distance_update_fp64
