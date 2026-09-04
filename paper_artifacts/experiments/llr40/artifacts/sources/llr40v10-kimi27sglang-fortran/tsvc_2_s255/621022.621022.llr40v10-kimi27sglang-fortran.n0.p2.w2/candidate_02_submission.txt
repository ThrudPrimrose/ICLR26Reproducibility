module tsvc_2_s255
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s255_fp64(a, b, LEN_1D) bind(C, name="tsvc_2_s255_fp64")
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i, n
    real(c_double) :: third

    n = LEN_1D
    third = 0.333_c_double

    if (n >= 2_c_int64_t) then
      a(1) = (b(1) + b(n) + b(n-1)) * third
      if (n >= 3_c_int64_t) then
        a(2) = (b(2) + b(1) + b(n)) * third
        do i = 3_c_int64_t, n
          a(i) = (b(i) + b(i-1) + b(i-2)) * third
        end do
      end if
    end if
  end subroutine tsvc_2_s255_fp64
end module tsvc_2_s255
