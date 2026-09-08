subroutine tsvc_2_s3111_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(2)
  real(c_double), intent(inout) :: workspace(*)
  integer(c_int64_t) :: i
  real(c_double) :: sum_val

  sum_val = 0.0d0
  !$omp parallel do simd reduction(+:sum_val)
  do i = 1, len_1d
    if (a(i) > 0.0d0) sum_val = sum_val + a(i)
  end do
  b(1) = sum_val
end subroutine tsvc_2_s3111_fp64
