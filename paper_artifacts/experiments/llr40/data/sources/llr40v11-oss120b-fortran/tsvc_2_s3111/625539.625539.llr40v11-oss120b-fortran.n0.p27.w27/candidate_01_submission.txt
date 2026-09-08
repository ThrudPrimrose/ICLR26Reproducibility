module tsvc_2_s3111_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s3111_fp64")
    ! Arguments: a - input array, b - output array of at least size 1, LEN_1D - length of a.
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: b(*)
    integer(c_int64_t), value :: LEN_1D
    real(c_double) :: sum
    integer(c_int64_t) :: i

    sum = 0.0_c_double
    !$omp parallel do simd reduction(+:sum) schedule(static)
    do i = 1, LEN_1D
      if (a(i) > 0.0_c_double) then
        sum = sum + a(i)
      end if
    end do
    b(1) = sum
  end subroutine tsvc_2_s3111_fp64
end module tsvc_2_s3111_mod
