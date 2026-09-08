subroutine tsvc_2_s316_fp64(a, result, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: result(1)
  integer(c_int64_t) :: i
  real(c_double) :: x

  x = huge(0.0_c_double)
  !$omp parallel do reduction(min:x)
  do i = 1, LEN_1D
    if (a(i) < x) then
      x = a(i)
    end if
  end do
  result(1) = x
end subroutine tsvc_2_s316_fp64
