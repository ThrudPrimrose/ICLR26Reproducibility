subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s152_fp64")
  use iso_c_binding
  integer(c_int64_t), value :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  real(c_double), intent(in) :: d(len_1d)
  real(c_double), intent(in) :: e(len_1d)
  integer(c_int64_t) :: i

  ! The two elementwise loops share nothing between iterations, so they fuse
  ! into one pass: b(i) is written and immediately consumed at the same i.
  !$omp parallel do simd
  do i = 1, len_1d
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
end subroutine tsvc_2_s152_fp64
