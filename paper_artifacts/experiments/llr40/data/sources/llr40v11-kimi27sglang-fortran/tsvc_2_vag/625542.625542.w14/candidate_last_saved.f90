module tsvc_2_vag_m
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(c,name="tsvc_2_vag_fp64")
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), intent(in), value :: LEN_1D
    integer(c_int64_t) :: i
    !$omp parallel do default(none) shared(a,b,ip,LEN_1D) private(i) schedule(static)
    do i = 1, LEN_1D
      a(i) = b(ip(i))
    end do
    !$omp end parallel do
  end subroutine
end module
