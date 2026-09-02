subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s, workspace, workspace_size) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  integer(c_int64_t), value, intent(in) :: s
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer :: i
!$omp parallel do default(none) schedule(static) shared(a,b,len_1d,s) private(i)
  do i = 1, len_1d
    a(i) = a(i) + b(i) * real(s, c_double)
  end do
!$omp end parallel do
end subroutine tsvc_2_vpvts_fp64
