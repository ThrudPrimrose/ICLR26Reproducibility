module tsvc_2_s4112_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_s4112_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    !$omp parallel do schedule(static) default(none) shared(a,b,ip,LEN_1D) private(i)
    do i = 1_c_int64_t, LEN_1D
      a(i) = a(i) + b(ip(i) + 1_c_int64_t) * 2.0_c_double
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s4112_fp64
end module tsvc_2_s4112_mod
