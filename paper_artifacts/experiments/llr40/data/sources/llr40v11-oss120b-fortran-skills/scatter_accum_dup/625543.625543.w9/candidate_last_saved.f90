subroutine scatter_accum_dup_fp64(bins, src, ip, LEN_1D, workspace, workspace_size) bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: bins(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do private(i)
  do i = 1, LEN_1D
    !$omp atomic
    bins(ip(i) + 1) = bins(ip(i) + 1) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
