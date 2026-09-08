subroutine compact_threshold_pack_fp64(out_count, src, weight, packed, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(out) :: out_count(1)
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: src(len_1d)
  real(c_double), intent(in) :: weight(len_1d)
  real(c_double), intent(inout) :: packed(len_1d)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: n
  ! Compute number of positive elements
  n = count(src > 0.0_c_double, kind=c_int64_t)
  if (n > 0_c_int64_t) then
    packed(1:n) = pack(src*weight, src > 0.0_c_double)
  end if
  out_count(1) = n
end subroutine compact_threshold_pack_fp64
