subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3112_fp64")
  use iso_c_binding
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(LEN_1D)

  integer(c_int64_t) :: i, n, r
  real(c_double) :: sum

  sum = 0.0d0
  n = LEN_1D
  r = mod(n, 16_c_int64_t)

  if (n >= 16) then
    do i = 1_c_int64_t, n - r, 16_c_int64_t
      sum = sum + a(i)
      b(i) = sum
      sum = sum + a(i+1)
      b(i+1) = sum
      sum = sum + a(i+2)
      b(i+2) = sum
      sum = sum + a(i+3)
      b(i+3) = sum
      sum = sum + a(i+4)
      b(i+4) = sum
      sum = sum + a(i+5)
      b(i+5) = sum
      sum = sum + a(i+6)
      b(i+6) = sum
      sum = sum + a(i+7)
      b(i+7) = sum
      sum = sum + a(i+8)
      b(i+8) = sum
      sum = sum + a(i+9)
      b(i+9) = sum
      sum = sum + a(i+10)
      b(i+10) = sum
      sum = sum + a(i+11)
      b(i+11) = sum
      sum = sum + a(i+12)
      b(i+12) = sum
      sum = sum + a(i+13)
      b(i+13) = sum
      sum = sum + a(i+14)
      b(i+14) = sum
      sum = sum + a(i+15)
      b(i+15) = sum
    end do
  end if

  do i = n - r + 1_c_int64_t, n
    sum = sum + a(i)
    b(i) = sum
  end do

end subroutine tsvc_2_s3112_fp64
