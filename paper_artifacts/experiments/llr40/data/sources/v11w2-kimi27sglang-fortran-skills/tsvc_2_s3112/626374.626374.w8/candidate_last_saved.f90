subroutine tsvc_2_s3112(a, b, LEN_1D) bind(C, name='tsvc_2_s3112_fp64')
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: b(LEN_1D)
  integer(c_int64_t) :: n, i
  real(c_double) :: s

  n = LEN_1D
  if (n <= 0) return

  s = 0.0d0
  i = 1
  do while (i + 3 <= n)
    s = s + a(i)
    b(i) = s
    s = s + a(i + 1)
    b(i + 1) = s
    s = s + a(i + 2)
    b(i + 2) = s
    s = s + a(i + 3)
    b(i + 3) = s
    i = i + 4
  end do
  do while (i <= n)
    s = s + a(i)
    b(i) = s
    i = i + 1
  end do
end subroutine tsvc_2_s3112
