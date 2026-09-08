subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s319_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
  real(c_double), intent(in)    :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
  real(c_double) :: s0, s1, s2, s3, sum
  integer(c_int64_t) :: i, n4, rem

  s0 = 0.0_c_double
  s1 = 0.0_c_double
  s2 = 0.0_c_double
  s3 = 0.0_c_double
  n4 = LEN_1D - mod(LEN_1D, 4_c_int64_t)
  !GCC$ ivdep
  do i = 1, n4, 4
    a(i)   = c(i)   + d(i)
    s0     = s0 + a(i)
    b(i)   = c(i)   + e(i)
    s0     = s0 + b(i)

    a(i+1) = c(i+1) + d(i+1)
    s1     = s1 + a(i+1)
    b(i+1) = c(i+1) + e(i+1)
    s1     = s1 + b(i+1)

    a(i+2) = c(i+2) + d(i+2)
    s2     = s2 + a(i+2)
    b(i+2) = c(i+2) + e(i+2)
    s2     = s2 + b(i+2)

    a(i+3) = c(i+3) + d(i+3)
    s3     = s3 + a(i+3)
    b(i+3) = c(i+3) + e(i+3)
    s3     = s3 + b(i+3)
  end do

  rem = n4 + 1
  !GCC$ ivdep
  do i = rem, LEN_1D
    a(i) = c(i) + d(i)
    s0   = s0 + a(i)
    b(i) = c(i) + e(i)
    s0   = s0 + b(i)
  end do

  sum = ((s0 + s1) + s2) + s3
  b(1) = sum
end subroutine tsvc_2_s319_fp64
