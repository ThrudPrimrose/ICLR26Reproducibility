subroutine compact_threshold_pack_fp64(LEN_1D, src, weight, packed, out_count, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: src(LEN_1D)
  real(c_double), intent(in) :: weight(LEN_1D)
  real(c_double), intent(out) :: packed(LEN_1D)
  integer(c_int64_t), intent(out) :: out_count(1)
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int64_t) :: i
  integer(c_int64_t) :: n, inc

  n = 0
  do i = 1, LEN_1D
    if (src(i) > 0.0d0) then
      n = n + 1
      packed(n) = src(i) * weight(i)
    end if
  end do
  out_count(1) = n
end subroutine compact_threshold_pack_fp64
