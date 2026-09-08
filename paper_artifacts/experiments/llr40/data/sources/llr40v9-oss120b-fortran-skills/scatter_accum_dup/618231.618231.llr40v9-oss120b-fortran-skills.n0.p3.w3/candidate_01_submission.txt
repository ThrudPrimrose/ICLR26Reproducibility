subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none

  ! Arguments – scalars must be VALUE and appear after arrays.
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: bins(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  integer(c_int64_t) :: i

  !$omp parallel do default(none) shared(bins, src, ip, LEN_1D) private(i) schedule(static)
  do i = 1, LEN_1D
    !$omp atomic
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
  !$omp end parallel do

end subroutine scatter_accum_dup_fp64