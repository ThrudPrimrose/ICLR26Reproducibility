subroutine tsvc_2_vtvtv_fp64(a, b, c, LEN_1D) bind(c, name="tsvc_2_vtvtv_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int64_t) :: i

  !$omp parallel do simd schedule(static)
  do i = 1, LEN_1D
    a(i) = a(i) * b(i) * c(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vtvtv_fp64
