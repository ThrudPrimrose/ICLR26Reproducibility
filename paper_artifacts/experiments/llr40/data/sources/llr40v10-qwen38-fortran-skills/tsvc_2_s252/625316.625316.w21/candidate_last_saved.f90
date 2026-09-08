subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  integer(c_int64_t) :: i
  ! reference:  t=0; for i: s=b[i]*c[i]; a[i]=s+t; t=s
  !   => a[i] = b[i]*c[i] + b[i-1]*c[i-1]   (rotated scalar: t == s(i-1))
  if (len_1d == 0) return
  a(1) = b(1) * c(1)
  if (len_1d < 2) return
  !$omp parallel do simd schedule(static)
  do i = 2, len_1d
    a(i) = b(i) * c(i) + b(i - 1) * c(i - 1)
  end do
end subroutine tsvc_2_s252_fp64
