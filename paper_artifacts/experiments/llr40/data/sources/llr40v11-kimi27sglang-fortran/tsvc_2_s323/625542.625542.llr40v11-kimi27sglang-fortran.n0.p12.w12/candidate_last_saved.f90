subroutine tsvc_2_s323_fp64(a, b, c, d, e, LEN_1D) bind(c, name="tsvc_2_s323_fp64")
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  integer(c_int64_t) :: i, rem
  real(c_double) :: bi
  
  rem = LEN_1D - 1
  i = 2
  bi = b(1)
  
  do while (rem >= 4)
    a(i)   = bi + c(i)*d(i)
    bi     = a(i) + c(i)*e(i)
    b(i)   = bi
    a(i+1) = bi + c(i+1)*d(i+1)
    bi     = a(i+1) + c(i+1)*e(i+1)
    b(i+1) = bi
    a(i+2) = bi + c(i+2)*d(i+2)
    bi     = a(i+2) + c(i+2)*e(i+2)
    b(i+2) = bi
    a(i+3) = bi + c(i+3)*d(i+3)
    bi     = a(i+3) + c(i+3)*e(i+3)
    b(i+3) = bi
    i = i + 4
    rem = rem - 4
  end do
  
  do while (rem > 0)
    a(i) = bi + c(i)*d(i)
    bi   = a(i) + c(i)*e(i)
    b(i) = bi
    i = i + 1
    rem = rem - 1
  end do
end subroutine
