module tsvc_2_s3112_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3112_fp64(a, b, LEN_1D) bind(C, name='tsvc_2_s3112_fp64')
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: b(*)
    integer(c_int64_t), value :: LEN_1D
    integer :: i
    real(c_double) :: sum
    sum = 0.0_c_double
    do i = 1, LEN_1D
      sum = sum + a(i)
      b(i) = sum
    end do
  end subroutine tsvc_2_s3112_fp64
end module tsvc_2_s3112_mod
