subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)

  integer(c_int64_t) :: i
  real(c_double) :: x
  integer(c_int64_t) :: idx

  if (LEN_1D <= 0_c_int64_t) then
    out_value(1) = 0.0_c_double
    out_index(1) = -1_c_int64_t
    return
  end if

  ! Early detection of first positive infinity (largest possible value)
  do i = 1_c_int64_t, LEN_1D
    if (ieee_class(a(i)) == ieee_positive_inf) then
      out_value(1) = a(i)
      out_index(1) = i - 1_c_int64_t
      return
    end if
  end do

  x = a(1)
  idx = 0_c_int64_t

  if (ieee_is_nan(x)) then
    out_value(1) = x
    out_index(1) = 0_c_int64_t
    return
  end if

  if (.not. ieee_is_finite(x)) then
    if (x > 0.0_c_double) then
      out_value(1) = x
      out_index(1) = 0_c_int64_t
      return
    end if
    ! Negative infinity: continue scanning
  end if

  do i = 2_c_int64_t, LEN_1D
    ! Skip NaNs
    if (ieee_is_nan(a(i))) cycle
    ! Handle infinities
    if (.not. ieee_is_finite(a(i))) then
        ! a(i) is either +inf or -inf
        if (a(i) > 0.0_c_double) then
            ! Positive infinity is the maximum possible
            x = a(i)
            idx = i - 1_c_int64_t
            exit
        else
            ! Negative infinity is less than any finite number; ignore
            cycle
        end if
    end if
    if (a(i) > x) then
        x = a(i)
        idx = i - 1_c_int64_t
    end if
  end do

  out_value(1) = x
  out_index(1) = idx
end subroutine argmax_with_index_fp64
