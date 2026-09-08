subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(c, name="tsvc_2_s1244_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*), c(*)
  real(c_double), intent(out)   :: d(*)
  integer(c_int64_t), intent(in), value :: LEN_1D

  integer(c_int64_t) :: i

  do i = 1, LEN_1D - 1
     a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
  end do

  do i = 1, LEN_1D - 1
     d(i) = a(i) + a(i + 1)
  end do

end subroutine tsvc_2_s1244_fp64
