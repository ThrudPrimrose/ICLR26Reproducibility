subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d, workspace_size
  real(c_double), intent(inout) :: a(len_1d), b(len_1d)
  real(c_double), intent(in)    :: c(len_1d), d(len_1d), e(len_1d)
  character(kind=c_char), intent(inout) :: workspace(workspace_size)
  real(c_double) :: sum_val
  integer(c_int64_t) :: i

  sum_val = 0.0d0
  !$omp parallel do simd reduction(+:sum_val)
  do i = 1_c_int64_t, len_1d
    a(i) = c(i) + d(i)
    sum_val = sum_val + a(i)
    b(i) = c(i) + e(i)
    sum_val = sum_val + b(i)
  end do
  b(1) = sum_val
end subroutine tsvc_2_s319_fp64
