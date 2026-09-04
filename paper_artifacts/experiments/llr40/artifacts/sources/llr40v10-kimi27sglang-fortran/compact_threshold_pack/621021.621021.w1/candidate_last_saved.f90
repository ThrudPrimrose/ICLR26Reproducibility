module compact_threshold_pack_m
  use iso_c_binding
  implicit none
contains
  subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, LEN_1D, workspace, workspace_bytes) bind(c, name="compact_threshold_pack_fp64")
    integer(c_int64_t), intent(out) :: out_count
    real(c_double), intent(in) :: src(*), weight(*)
    real(c_double), intent(out) :: packed(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int8_t), intent(in) :: workspace(*)
    integer(c_int64_t), value, intent(in) :: workspace_bytes
    integer(c_int64_t) :: i, n
    n = 0
    do i = 1, LEN_1D
       if (src(i) > 0.0_c_double) then
          n = n + 1
          packed(n) = src(i) * weight(i)
       end if
    end do
    out_count = n
  end subroutine compact_threshold_pack_fp64
end module compact_threshold_pack_m
