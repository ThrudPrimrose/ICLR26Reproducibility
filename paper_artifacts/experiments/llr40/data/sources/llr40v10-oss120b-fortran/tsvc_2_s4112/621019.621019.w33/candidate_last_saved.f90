module tsvc_2_s4112_mod
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
    ! Arguments
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in)    :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    ! Locals
    integer :: i, idx
    ! Parallel loop
        do i = 0, int(LEN_1D, kind=4) - 1
      idx = ip(i+1) + 1
      a(i+1) = a(i+1) + b(idx) * 2.0_c_double
    end do
      end subroutine tsvc_2_s4112_fp64
end module tsvc_2_s4112_mod
