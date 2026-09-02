module tsvc_2_s4112_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(c, name='tsvc_2_s4112_fp64')
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i

    do i = 1, LEN_1D
      a(i) = a(i) + b(ip(i)) * 2.0_c_double
    end do
  end subroutine
end module
