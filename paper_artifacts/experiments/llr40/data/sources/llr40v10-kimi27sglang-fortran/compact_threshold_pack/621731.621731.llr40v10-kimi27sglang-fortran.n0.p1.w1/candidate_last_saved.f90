subroutine compact_threshold_pack_fp64(out_count, packed, src, weight, LEN_1D, workspace, workspace_bytes) bind(c, name="compact_threshold_pack_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_signed_char
  implicit none
  integer(c_int64_t), dimension(*), intent(out) :: out_count
  real(c_double), dimension(*), intent(out) :: packed
  real(c_double), dimension(*), intent(in) :: src, weight
  integer(c_int64_t), value :: LEN_1D
  integer(c_signed_char), dimension(*), intent(inout) :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i, n
  n = 0
  do i = 1, LEN_1D
    if (src(i) > 0.0_c_double) then
      n = n + 1
      packed(n) = src(i) * weight(i)
    end if
  end do
  out_count(1) = n
end subroutine compact_threshold_pack_fp64
