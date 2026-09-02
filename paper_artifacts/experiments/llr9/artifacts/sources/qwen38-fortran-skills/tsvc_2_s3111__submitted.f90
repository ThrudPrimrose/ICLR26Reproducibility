subroutine tsvc_2_s3111_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in)    :: a(len_1d)
  real(c_double), intent(out)   :: b(2)

  integer(c_int64_t) :: i, n
  real(c_double) :: s

  n = len_1d
  s = 0.0d0
  !$omp parallel do simd reduction(+:s)
  do i = 1, n
    s = s + merge(a(i), 0.0d0, a(i) > 0.0d0)
  end do
  b(1) = s
end subroutine tsvc_2_s3111_fp64
