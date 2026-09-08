subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  real(c_double), intent(inout) :: d(LEN_1D)
  integer(c_int8_t), target, intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i

  !$omp simd
  do i = 1, LEN_1D - 1
    a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    d(i) = a(i) + a(i + 1)
  end do
end subroutine tsvc_2_s1244_fp64
