subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t) :: i

  !$omp parallel do simd schedule(guided)
  do i = 1, LEN_1D
    a(i) = b(ip(i))
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vag_fp64
