subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C, name="tsvc_2_s316_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)

  real(c_double) :: x
  integer(c_int64_t) :: i

  if (len_1d <= 0) then
    result(1) = 0.0d0
    return
  end if
  x = a(1)
  if (len_1d >= 2) then
    !$omp parallel do simd reduction(min:x)
    do i = 2, len_1d
      if (a(i) < x) x = a(i)
    end do
  end if
  result(1) = x
end subroutine tsvc_2_s316_fp64
