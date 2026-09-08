module tsvc_2_vtvtv_mod
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
contains
  subroutine tsvc_2_vtvtv_fp64(a, b, c, LEN_1D) bind(C)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: c(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t) :: i
    !$omp parallel do schedule(static) default(none) shared(a,b,c,LEN_1D) private(i)
    do i = 1, LEN_1D
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_vtvtv_fp64
end module tsvc_2_vtvtv_mod
