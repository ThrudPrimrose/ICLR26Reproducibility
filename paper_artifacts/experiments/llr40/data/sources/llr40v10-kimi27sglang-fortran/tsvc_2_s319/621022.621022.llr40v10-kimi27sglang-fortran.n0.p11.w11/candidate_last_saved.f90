module tsvc_2_s319_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(c, name='tsvc_2_s319_fp64')
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(out) :: a(LEN_1D), b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
    real(c_double) :: sum_a, sum_b
    integer(c_int64_t) :: i
    sum_a = 0.0_c_double
    sum_b = 0.0_c_double
    !GCC$ unroll 8
    do i = 1, LEN_1D
      a(i) = c(i) + d(i)
      sum_a = sum_a + a(i)
      b(i) = c(i) + e(i)
      sum_b = sum_b + b(i)
    end do
    b(1) = sum_a + sum_b
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
