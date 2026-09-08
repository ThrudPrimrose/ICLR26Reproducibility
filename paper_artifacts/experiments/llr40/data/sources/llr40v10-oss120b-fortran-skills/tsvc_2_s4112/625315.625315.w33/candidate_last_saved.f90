module tsvc_2_s4112_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C)
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i
    
    !$omp parallel default(none) shared(a,b,ip,LEN_1D)
    !$omp do schedule(static)
    do i = 1, LEN_1D
      a(i) = a(i) + b(ip(i)) * 2.0d0
    end do
    !$omp end do
    !$omp end parallel
  end subroutine tsvc_2_s4112_fp64
end module tsvc_2_s4112_mod
