subroutine tsvc_2_s311_fp64(a, sum_out, LEN_1D, workspace, workspace_bytes) bind(C, name="tsvc_2_s311_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)  :: a(*)
  real(c_double), intent(out) :: sum_out(*)
  integer(c_int64_t), value   :: LEN_1D
  type(c_ptr),       value   :: workspace
  integer(c_int64_t), value  :: workspace_bytes
  real(c_double) :: s
  integer(c_int64_t) :: i

  s = 0.0_c_double
  !$omp parallel do reduction(+:s) schedule(static)
  do i = 1, LEN_1D
     s = s + a(i)
  end do
  !$omp end parallel do
  sum_out(1) = s
end subroutine tsvc_2_s311_fp64
