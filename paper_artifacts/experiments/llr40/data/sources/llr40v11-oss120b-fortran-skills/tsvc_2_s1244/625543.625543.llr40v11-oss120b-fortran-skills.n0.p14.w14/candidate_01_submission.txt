module tsvc_2_s1244_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(in)    :: c(*)
    real(c_double), intent(inout) :: d(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: a_new, tmp
    do i = 0, LEN_1D - 2
      tmp = a(i+2)
      a_new = b(i+1) + c(i+1)*c(i+1) + b(i+1)*b(i+1) + c(i+1)
      a(i+1) = a_new
      d(i+1) = a_new + tmp
    end do
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_mod
