subroutine tsvc_2_s316_fp64(a, result, lenarg) bind(C, name="tsvc_2_s316_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in)           :: a(0:2147483647)
  real(c_double), intent(out)          :: result(0:0)
  real(c_double), intent(in), target   :: lenarg(0:0)
  type(c_ptr) :: p
  integer(c_int64_t) :: n, i
  real(c_double) :: x

  p = c_loc(lenarg)
  n = transfer(p, n)   ! n = LEN_1D (the int64 C passed by value)

  x = a(0)
  do i = 1, n - 1
    if (a(i) < x) x = a(i)
  end do
  result(0) = x
end subroutine tsvc_2_s316_fp64
