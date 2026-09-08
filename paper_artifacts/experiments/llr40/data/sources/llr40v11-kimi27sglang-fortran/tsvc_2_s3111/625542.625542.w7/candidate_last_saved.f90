module tsvc_2_s3111_mod
  use, intrinsic :: iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s3111_fp64(a, b, LEN_1D) bind(C, name='tsvc_2_s3111_fp64')
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: s
    integer(c_int64_t) :: i
    s = 0.0_c_double
    do i = 1, LEN_1D
      s = s + max(a(i), 0.0_c_double)
    end do
    b(1) = s
  end subroutine tsvc_2_s3111_fp64
end module tsvc_2_s3111_mod
