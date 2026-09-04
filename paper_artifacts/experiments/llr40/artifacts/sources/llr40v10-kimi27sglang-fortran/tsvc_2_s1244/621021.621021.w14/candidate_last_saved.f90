module tsvc_2_s1244_m
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(c)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(inout) :: d(*)
    integer(c_int64_t), intent(in), value :: LEN_1D

    integer(c_int64_t) :: i, n

    n = LEN_1D - 1

    !$omp parallel do simd
    do i = 1, n
      d(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i) + a(i + 1)
    end do
    !$omp end parallel do simd

    !$omp parallel do simd
    do i = 1, n
      a(i) = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_s1244_fp64
end module tsvc_2_s1244_m
