subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, S, workspace_size
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t) :: i

!$omp parallel do simd schedule(static) default(none) shared(a, b, LEN_1D, S, workspace, workspace_size)
  do i = 1, LEN_1D
    a(i) = a(i) + b(i) * S
  end do
!$omp end parallel do simd
end subroutine tsvc_2_vpvts_fp64
