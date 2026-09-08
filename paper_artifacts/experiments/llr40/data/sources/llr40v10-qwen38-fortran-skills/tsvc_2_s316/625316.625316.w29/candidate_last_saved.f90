subroutine tsvc_2_s316_fp64(a, result, len_1d) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: result(1)

  real(c_double) :: x
  integer(c_int64_t) :: i

  x = a(1)
  ! pure min reduction: no carried state, unit stride, independent iterations
  !$omp parallel do simd reduction(min:x)
  do i = 2, len_1d
    if (a(i) < x) x = a(i)
  end do
  result(1) = x
end subroutine tsvc_2_s316_fp64
