subroutine tsvc_2_vtvtv_fp64(a, b, c, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)

  integer :: i, n

  n = int(LEN_1D)
  !$omp parallel do simd schedule(static) simdlen(8)
  do i = 1, n
    a(i) = a(i) * b(i) * c(i)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vtvtv_fp64
