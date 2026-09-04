module tsvc_2_s152_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, LEN_1D) bind(c)
    real(c_double), intent(inout) :: a(*), b(*)
    real(c_double), intent(in) :: c(*), d(*), e(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i

    !$omp parallel do simd schedule(static)
    do i = 1, LEN_1D
      b(i) = d(i) * e(i)
    end do
    !$omp parallel do simd schedule(static)
    do i = 1, LEN_1D
      a(i) = a(i) + b(i) * c(i)
    end do
  end subroutine
end module
