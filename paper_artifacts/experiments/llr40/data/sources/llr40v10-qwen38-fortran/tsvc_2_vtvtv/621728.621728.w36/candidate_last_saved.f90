subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(c, name="tsvc_2_vtvtv_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  integer(c_int64_t) :: i
  !$omp parallel do schedule(static)
  do i = 1, len_1d
    a(i) = a(i) * b(i) * c(i)
  end do
  !$omp end parallel do
end subroutine tsvc_2_vtvtv_fp64
