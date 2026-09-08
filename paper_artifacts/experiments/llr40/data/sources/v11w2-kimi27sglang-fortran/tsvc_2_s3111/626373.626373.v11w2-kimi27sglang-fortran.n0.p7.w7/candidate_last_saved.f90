module tsvc_2_s3111_m
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(c)
    integer(c_int64_t), intent(in), value :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: b(*)
    real(c_double) :: sum
    integer(c_int64_t) :: i

    sum = 0.0_c_double
    !$omp parallel do reduction(+:sum) schedule(guided, 4096)
    do i = 1, LEN_1D
      sum = sum + merge(a(i), 0.0_c_double, a(i) > 0.0_c_double)
    end do
    b(1) = sum
  end subroutine tsvc_2_s3111_fp64
end module tsvc_2_s3111_m
