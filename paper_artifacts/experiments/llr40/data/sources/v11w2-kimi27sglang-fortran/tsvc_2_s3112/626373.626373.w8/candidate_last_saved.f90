module tsvc_2_s3112
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name='tsvc_2_s3112_fp64')
    integer(c_int64_t), value :: LEN_1D
    real(c_double), dimension(LEN_1D), intent(in) :: a
    real(c_double), dimension(LEN_1D), intent(out) :: b
    integer(c_int64_t) :: i
    real(c_double) :: sum

    sum = 0.0_c_double
    !$omp simd simdlen(2) reduction(inscan, +:sum)
    do i = 1, LEN_1D
      sum = sum + a(i)
      !$omp scan inclusive(sum)
      b(i) = sum
    end do
    !$omp end simd
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112
