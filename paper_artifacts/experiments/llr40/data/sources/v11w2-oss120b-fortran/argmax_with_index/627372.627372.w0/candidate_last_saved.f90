subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding, only: C_DOUBLE, C_INT64_T
  use, intrinsic :: ieee_arithmetic
  implicit none
  real(C_DOUBLE), intent(in) :: a(*)
  integer(C_INT64_T), intent(out) :: out_index(1)
  real(C_DOUBLE), intent(out) :: out_value(1)
  integer(C_INT64_T), value :: LEN_1D
  integer(C_INT64_T) :: i, idx_c
  real(C_DOUBLE) :: x

  if (LEN_1D <= 0_C_INT64_T) then
    out_value(1) = 0.0_C_DOUBLE
    out_index(1) = 0_C_INT64_T
    return
  end if

  if (ieee_is_nan(a(1))) then
    out_value(1) = a(1)
    out_index(1) = 0_C_INT64_T
    return
  end if

  x = a(1)
  idx_c = 0_C_INT64_T
  do i = 2, LEN_1D
    if (a(i) > x) then
      x = a(i)
      idx_c = i - 1_C_INT64_T
    end if
  end do
  out_value(1) = x
  out_index(1) = idx_c
end subroutine argmax_with_index_fp64
