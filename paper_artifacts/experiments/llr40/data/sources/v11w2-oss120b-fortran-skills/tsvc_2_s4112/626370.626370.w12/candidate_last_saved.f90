module tsvc_2_s4112_mod
  use iso_c_binding, only: c_int64_t, c_int32_t, c_double
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C)
    ! Arguments from C
    real(c_double), intent(inout), volatile :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, idx
        do i = 1, LEN_1D
      idx = int(ip(i), kind=c_int64_t) + 1_c_int64_t
      a(i) = a(i) + b(idx) * 2.0d0
    end do
  end subroutine tsvc_2_s4112_fp64
end module tsvc_2_s4112_mod
