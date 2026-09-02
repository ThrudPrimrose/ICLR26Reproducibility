subroutine fuse_stencil_through_transient_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C, name="fuse_stencil_through_transient_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(inout) :: out(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 2_c_int64_t, LEN_1D - 2_c_int64_t
     out(i) = (a(i - 1_c_int64_t) + a(i) + a(i + 1_c_int64_t)) &
          & * (a(i) + a(i + 1_c_int64_t) + a(i + 2_c_int64_t))
  end do
  !$omp end parallel do simd
end subroutine fuse_stencil_through_transient_fp64
