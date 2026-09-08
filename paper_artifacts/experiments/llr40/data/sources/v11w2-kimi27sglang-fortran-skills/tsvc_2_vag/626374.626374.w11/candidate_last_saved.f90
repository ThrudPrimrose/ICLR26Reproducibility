subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D, workspace, workspace_size) bind(C, name='tsvc_2_vag_fp64')
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t, c_int8_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  integer(c_int8_t), intent(in) :: workspace(*)
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in)  :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do proc_bind(spread) schedule(guided, 1920)
  do i = 1, LEN_1D
     a(i) = b(ip(i))
  end do
  !$omp end parallel do
end subroutine tsvc_2_vag_fp64
