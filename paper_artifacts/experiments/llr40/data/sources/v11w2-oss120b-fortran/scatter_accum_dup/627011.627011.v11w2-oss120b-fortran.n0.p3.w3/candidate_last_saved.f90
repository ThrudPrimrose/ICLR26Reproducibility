subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: bins(*)
  integer(c_int32_t), intent(in) :: ip(*)
  real(c_double), intent(in) :: src(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i
  !$omp parallel do default(none) shared(bins, ip, src, LEN_1D) private(i)
  do i = 1, LEN_1D
    !$omp atomic
    bins(ip(i)) = bins(ip(i)) + src(i)
  end do
  !$omp end parallel do
end subroutine scatter_accum_dup_fp64
