subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: bins(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  integer(c_int8_t), intent(inout) :: workspace(*)
  integer(c_int64_t) :: i, j, n12

  n12 = (LEN_1D / 12) * 12

  !$omp parallel do schedule(static)
  do i = 1, n12, 12
    !$omp atomic update relaxed
    bins(ip(i)) = bins(ip(i)) + src(i)
    !$omp atomic update relaxed
    bins(ip(i+1)) = bins(ip(i+1)) + src(i+1)
    !$omp atomic update relaxed
    bins(ip(i+2)) = bins(ip(i+2)) + src(i+2)
    !$omp atomic update relaxed
    bins(ip(i+3)) = bins(ip(i+3)) + src(i+3)
    !$omp atomic update relaxed
    bins(ip(i+4)) = bins(ip(i+4)) + src(i+4)
    !$omp atomic update relaxed
    bins(ip(i+5)) = bins(ip(i+5)) + src(i+5)
    !$omp atomic update relaxed
    bins(ip(i+6)) = bins(ip(i+6)) + src(i+6)
    !$omp atomic update relaxed
    bins(ip(i+7)) = bins(ip(i+7)) + src(i+7)
    !$omp atomic update relaxed
    bins(ip(i+8)) = bins(ip(i+8)) + src(i+8)
    !$omp atomic update relaxed
    bins(ip(i+9)) = bins(ip(i+9)) + src(i+9)
    !$omp atomic update relaxed
    bins(ip(i+10)) = bins(ip(i+10)) + src(i+10)
    !$omp atomic update relaxed
    bins(ip(i+11)) = bins(ip(i+11)) + src(i+11)
  end do
  !$omp end parallel do

  do j = n12 + 1, LEN_1D
    !$omp atomic update relaxed
    bins(ip(j)) = bins(ip(j)) + src(j)
  end do
end subroutine scatter_accum_dup_fp64
