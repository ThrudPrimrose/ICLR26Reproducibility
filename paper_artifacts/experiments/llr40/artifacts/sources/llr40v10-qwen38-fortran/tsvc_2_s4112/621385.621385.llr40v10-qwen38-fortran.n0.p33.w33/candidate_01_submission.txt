subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d) bind(c, name="tsvc_2_s4112_fp64")
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  real(c_double), intent(inout)   :: a(*)
  real(c_double), intent(in)      :: b(*)
  integer(c_int32_t), intent(in)  :: ip(*)
  integer(c_int64_t), intent(in), value :: len_1d
  integer(c_int64_t) :: i

  !$omp parallel do
  do i = 1, len_1d
     a(i) = a(i) + 2.0d0 * b(ip(i))
  end do
  !$omp end parallel do
end subroutine tsvc_2_s4112_fp64
