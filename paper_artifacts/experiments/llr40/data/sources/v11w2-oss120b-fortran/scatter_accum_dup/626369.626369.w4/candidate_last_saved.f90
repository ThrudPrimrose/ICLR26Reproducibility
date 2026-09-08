module scatter_accum_dup_mod
  use iso_c_binding
  implicit none
contains
  subroutine scatter_accum_dup_fp64(bins, src, ip, LEN_1D, workspace, workspace_bytes) bind(C, name="scatter_accum_dup_fp64")
    real(c_double), intent(inout) :: bins(*)
    real(c_double), intent(in) :: src(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer(c_int64_t) :: i, idx
    do i = 1, LEN_1D
      idx = ip(i)
      if (idx < 0_c_int64_t) idx = idx + LEN_1D
      bins(idx + 1) = bins(idx + 1) + src(i)
    end do
  end subroutine scatter_accum_dup_fp64
end module scatter_accum_dup_mod
