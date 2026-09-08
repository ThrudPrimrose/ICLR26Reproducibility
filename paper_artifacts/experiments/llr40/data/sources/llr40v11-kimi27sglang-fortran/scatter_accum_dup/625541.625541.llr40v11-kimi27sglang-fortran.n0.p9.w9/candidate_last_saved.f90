module scatter_accum_dup_mod
  use iso_c_binding
  implicit none
contains
  subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D, workspace, workspace_size) &
      bind(c, name="scatter_accum_dup_fp64")
    integer(c_int64_t), value      :: LEN_1D
    integer(c_int64_t), value      :: workspace_size
    real(c_double), intent(inout)  :: bins(LEN_1D)
    integer(c_int32_t), intent(in)  :: ip(LEN_1D)
    real(c_double), intent(in)     :: src(LEN_1D)
    integer(c_int8_t), intent(inout) :: workspace(workspace_size)

    integer(c_int64_t) :: i, n, idx

    n = LEN_1D
    if (n <= 0) return

    !$omp parallel do schedule(dynamic, 1024) private(idx)
    do i = 1, n
      idx = ip(i)
      !$omp atomic update
      bins(idx) = bins(idx) + src(i)
    end do
    !$omp end parallel do
  end subroutine scatter_accum_dup_fp64
end module scatter_accum_dup_mod
