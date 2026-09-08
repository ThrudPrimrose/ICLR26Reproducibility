subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(C, name="tsvc_2_s252_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_1D
  real(c_double), intent(out) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
  integer(c_int64_t) :: i

  a(1) = b(1) * c(1)
  !$omp parallel do simd if(LEN_1D > 2048)
  do i = 2, LEN_1D
    a(i) = b(i) * c(i) + b(i-1) * c(i-1)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_s252_fp64
