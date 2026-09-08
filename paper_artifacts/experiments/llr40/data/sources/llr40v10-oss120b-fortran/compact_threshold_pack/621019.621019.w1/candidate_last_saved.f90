subroutine compact_threshold_pack_fp64(src, weight, packed, out_count, LEN_1D) bind(C, name="compact_threshold_pack_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(in) :: src(*)
  real(c_double), intent(in) :: weight(*)
  real(c_double), intent(out) :: packed(*)
  integer(c_int64_t), intent(out) :: out_count(1)
  integer(c_int64_t) :: i, n
  n = 0_c_int64_t
  do i = 1, LEN_1D
    if (src(i) > 0.0_c_double) then
      packed(n+1) = src(i) * weight(i)
      n = n + 1_c_int64_t
    end if
  end do
  out_count(1) = n
end subroutine compact_threshold_pack_fp64
