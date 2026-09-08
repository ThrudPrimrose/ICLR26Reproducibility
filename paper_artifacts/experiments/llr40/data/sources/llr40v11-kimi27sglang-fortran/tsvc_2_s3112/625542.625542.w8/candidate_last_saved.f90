module tsvc_2_s3112_fp64_mod
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(c, name="tsvc_2_s3112_fp64")
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: b(LEN_1D)
    real(c_double) :: sum
    integer(c_int64_t) :: i

    sum = 0.0_c_double
    !$omp simd reduction(inscan, +:sum)
    do i = 1, LEN_1D
      sum = sum + a(i)
      !$omp scan inclusive(sum)
      b(i) = sum
    end do
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112_fp64_mod
