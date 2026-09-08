subroutine compact_threshold_pack(src, weight, packed, out_count, LEN_1D) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: src(LEN_1D)
  real(c_double), intent(in) :: weight(LEN_1D)
  real(c_double), intent(inout) :: packed(LEN_1D)
  integer(c_int64_t), intent(out) :: out_count(1)

  integer(c_int64_t) :: i, n
  n = 0
  do i = 1, LEN_1D
    if (src(i) > 0.0d0) then
      packed(n+1) = src(i) * weight(i)
      n = n + 1
    end if
  end do
  out_count(1) = n
end subroutine compact_threshold_pack
