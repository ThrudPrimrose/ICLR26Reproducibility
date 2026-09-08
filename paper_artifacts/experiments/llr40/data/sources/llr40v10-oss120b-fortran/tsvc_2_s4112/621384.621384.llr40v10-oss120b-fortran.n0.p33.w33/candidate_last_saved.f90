module tsvc_2_s4112_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i, ip_idx
    !$omp parallel do simd shared(a,b,ip) private(i, ip_idx)
    do i = 0, LEN_1D - 1
      ip_idx = int(ip(i+1), kind=c_int64_t)
      a(i+1) = a(i+1) + b(ip_idx) * 2.0_c_double
    end do
  end subroutine tsvc_2_s4112_fp64
end module tsvc_2_s4112_mod
