subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  integer(c_int64_t), intent(out) :: out_index(1)
  real(c_double), intent(out) :: out_value(1)

  integer(c_int64_t) :: i
  real(c_double) :: m
  integer(c_int64_t) :: first

  if (LEN_1D <= 1) then
    out_value(1) = a(1)
    out_index(1) = 0
    return
  end if

  m = -huge(0.0d0)
  first = LEN_1D + 1

  if (LEN_1D < 1048576) then
    do i = 1, LEN_1D
      m = max(m, a(i))
    end do
    do i = 1, LEN_1D
      if (a(i) == m .and. i < first) first = i
    end do
  else
    !$omp parallel do simd reduction(max:m)
    do i = 1, LEN_1D
      m = max(m, a(i))
    end do
    !$omp parallel do simd reduction(min:first)
    do i = 1, LEN_1D
      if (a(i) == m .and. i < first) first = i
    end do
  end if

  if (first > LEN_1D) first = 1
  out_value(1) = m
  out_index(1) = first - 1
end subroutine argmax_with_index_fp64
