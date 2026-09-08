module tsvc_2_s319_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s319_fp64")
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(out) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    real(c_double) :: sum
    real(c_double) :: c_val, a_val, b_val
    sum = 0.0_c_double
    !$omp parallel do reduction(+:sum) schedule(static) private(i,c_val,a_val,b_val)
    do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
      c_val = c(i+1)
      a_val = c_val + d(i+1)
      b_val = c_val + e(i+1)
      a(i+1) = a_val
      b(i+1) = b_val
      sum = sum + a_val + b_val
    end do
    !$omp end parallel do
    b(1) = sum
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
