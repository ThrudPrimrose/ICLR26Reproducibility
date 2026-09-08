subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(c, name="tsvc_2_vpvts_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in)    :: b(*)
  integer(c_int64_t), value     :: len_1d
  integer(c_int64_t), value     :: s
  real(c_double) :: sd
  integer(c_int64_t) :: i

  sd = real(s, c_double)
  !$omp parallel do schedule(static)
  do i = 1, len_1d
     a(i) = a(i) + b(i) * sd
  end do
  !$omp end parallel do
end subroutine tsvc_2_vpvts_fp64
