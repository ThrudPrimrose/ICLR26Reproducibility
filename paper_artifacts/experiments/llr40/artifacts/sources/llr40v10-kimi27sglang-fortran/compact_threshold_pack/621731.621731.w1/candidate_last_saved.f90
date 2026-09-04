subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, LEN_1D, workspace, workspace_bytes) bind(c, name="compact_threshold_pack_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_signed_char
  implicit none
  integer(c_int64_t), dimension(*), intent(out) :: out_count
  real(c_double), dimension(*), intent(in) :: src, weight
  real(c_double), dimension(*), intent(out) :: packed
  integer(c_int64_t), value :: LEN_1D
  integer(c_signed_char), dimension(*), intent(inout) :: workspace
  integer(c_int64_t), value :: workspace_bytes
  integer(c_int64_t) :: i
  print *, 'DIAG src(1:4):', src(1), src(2), src(3), src(4)
  print *, 'DIAG weight(1:4):', weight(1), weight(2), weight(3), weight(4)
  print *, 'DIAG packed(1:4):', packed(1), packed(2), packed(3), packed(4)
  print *, 'LEN_1D:', LEN_1D
  out_count(1) = 0
  do i = 1, LEN_1D
    if (src(i) > 0.0_c_double) then
      out_count(1) = out_count(1) + 1
    end if
  end do
  call flush(6)
end subroutine compact_threshold_pack_fp64
