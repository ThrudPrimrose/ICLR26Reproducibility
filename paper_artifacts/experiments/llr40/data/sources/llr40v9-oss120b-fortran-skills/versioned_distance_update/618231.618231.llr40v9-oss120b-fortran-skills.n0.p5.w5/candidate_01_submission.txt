subroutine versioned_distance_update_fp64(a, b, c, k, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: k, len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d), c(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i0, i

  !$omp parallel do schedule(static) private(i)
  do i0 = 1, k
    do i = i0 + k, len_1d, k
      a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
    end do
  end do
  !$omp end parallel do

end subroutine versioned_distance_update_fp64