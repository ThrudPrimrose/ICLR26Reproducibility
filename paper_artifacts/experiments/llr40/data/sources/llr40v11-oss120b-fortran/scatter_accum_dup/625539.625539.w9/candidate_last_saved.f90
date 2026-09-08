subroutine scatter_accum_dup_fp64(bins, src, ip, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  use, intrinsic :: ieee_arithmetic
  real(c_double), intent(inout) :: bins(*)
  real(c_double), intent(in) :: src(*)
  integer(c_int32_t), intent(in) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  integer(c_int64_t) :: i, idx
  real(c_double), allocatable :: src_local(:)
  if (LEN_1D <= 0) return
  allocate(src_local(LEN_1D))
  do i = 1_c_int64_t, LEN_1D
      src_local(i) = src(i)
  end do
  do i = 1_c_int64_t, LEN_1D
    idx = int(ip(i), c_int64_t)
    if (idx >= 0_c_int64_t) then
        if (idx < LEN_1D) then
            idx = idx + 1_c_int64_t
        else
            cycle
        end if
    else
        if (idx >= -LEN_1D) then
            idx = LEN_1D + idx + 1_c_int64_t
        else
            cycle
        end if
    end if
    ! NaN-aware accumulation
    if (ieee_is_nan(bins(idx))) then
        bins(idx) = src_local(i)
    else
        bins(idx) = bins(idx) + src_local(i)
    end if
  end do
  deallocate(src_local)
end subroutine scatter_accum_dup_fp64
