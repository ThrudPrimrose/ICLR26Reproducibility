subroutine tsvc_2_s3112_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(len_1d)
  real(c_double), intent(out) :: b(len_1d)
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i
  real(c_double) :: s

  s = 0.0d0
  !$omp parallel do simd reduction(inscan, +:s)
  do i = 1, len_1d
    s = s + a(i)
    !$omp scan inclusive(s)
    b(i) = s
  end do
end subroutine tsvc_2_s3112_fp64
