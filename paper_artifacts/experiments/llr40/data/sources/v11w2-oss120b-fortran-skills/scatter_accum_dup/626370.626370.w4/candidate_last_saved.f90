module scatter_accum_dup_mod
  use iso_c_binding, only: c_int64_t, c_double, c_int32_t, c_ptr
  implicit none
contains
  subroutine scatter_accum_dup_fp64(bins, src, ip, LEN_1D, workspace, workspace_size) bind(C, name="scatter_accum_dup_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: bins(LEN_1D)
    real(c_double), intent(in) :: src(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i
    integer(c_int64_t) :: idx
        !$omp parallel do schedule(static) default(none) private(i, idx) shared(bins, src, ip, LEN_1D)
      do i = 1, LEN_1D
      idx = ip(i) + 1
            !$omp atomic
      bins(idx) = bins(idx) + src(i)
          end do
!$omp end parallel do
      end subroutine scatter_accum_dup_fp64
end module scatter_accum_dup_mod
