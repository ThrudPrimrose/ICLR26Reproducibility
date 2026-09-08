subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s152_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(:), b(:)
  real(c_double), intent(in)    :: c(:), d(:), e(:)
  integer(c_int64_t), intent(in) :: len_1d
  integer(c_int64_t) :: i

  !$omp parallel do
  do i = 1, len_1d
     b(i) = d(i) * e(i)
     a(i) = a(i) + b(i) * c(i)
  end do
end subroutine tsvc_2_s152_fp64
