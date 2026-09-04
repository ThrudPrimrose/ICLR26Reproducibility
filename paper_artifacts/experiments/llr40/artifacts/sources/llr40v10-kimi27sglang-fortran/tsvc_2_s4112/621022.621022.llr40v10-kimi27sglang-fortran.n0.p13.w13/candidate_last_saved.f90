module tsvc_2_s4112
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(c, name="tsvc_2_s4112_fp64")
    real(c_double), dimension(*), intent(inout) :: a
    real(c_double), dimension(*), intent(in) :: b
    integer(c_int32_t), dimension(*), intent(in) :: ip
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i

    !$omp simd
    do i = 1, LEN_1D
      a(i) = a(i) + b(ip(i)) * 2.0_c_double
    end do
    !$omp end simd
  end subroutine
end module
