subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do
  do i = 1, LEN_1D
    a(i) = a(i) + b(ip(i)) * 2.0d0
  end do
  !$omp end parallel do
end subroutine tsvc_2_s4112_fp64
