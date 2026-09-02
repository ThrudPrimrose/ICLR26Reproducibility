subroutine tsvc_2_vpvts_fp64(a, b, LEN_1D, S_int, workspace, workspace_bytes) bind(C, name="tsvc_2_vpvts_fp64")
  use iso_c_binding, only: c_int64_t, c_double, c_ptr
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int64_t), value :: S_int
  type(c_ptr), value :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i

  !$omp parallel do default(none) shared(a,b,S_int,LEN_1D) private(i)
  do i = 1, LEN_1D
    a(i) = a(i) + b(i) * real(S_int, c_double)
  end do
  !$omp end parallel do
end subroutine tsvc_2_vpvts_fp64
