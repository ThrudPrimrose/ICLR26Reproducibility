subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D, workspace_size
  integer(c_int64_t), intent(out) :: out_count(1)
  real(c_double), intent(out) :: packed(LEN_1D)
  real(c_double), intent(in) :: src(LEN_1D)
  real(c_double), intent(in) :: weight(LEN_1D)
  integer(c_int8_t), intent(out) :: workspace(workspace_size)
  integer(c_int64_t) :: i, n
  n = 0
  do i = 1, LEN_1D
    if (src(i) > 0.0d0) then
      n = n + 1
      packed(n) = src(i) * weight(i)
    end if
  end do
  out_count(1) = n
end subroutine compact_threshold_pack_fp64
