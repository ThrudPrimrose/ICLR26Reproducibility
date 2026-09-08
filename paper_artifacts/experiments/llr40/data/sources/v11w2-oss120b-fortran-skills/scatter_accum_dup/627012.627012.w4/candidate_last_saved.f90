subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: bins(*)
  integer(c_int32_t), intent(in) :: ip(*)
  real(c_double), intent(in) :: src(*)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i

    !$omp parallel do schedule(static)
  do i = 1, LEN_1D
    !$omp atomic
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
