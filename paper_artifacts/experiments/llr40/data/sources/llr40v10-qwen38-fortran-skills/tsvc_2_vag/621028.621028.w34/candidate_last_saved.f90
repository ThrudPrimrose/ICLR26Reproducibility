subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(out) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int32_t), intent(in) :: ip(len_1d)
  integer(c_int64_t) :: i

  !$omp parallel do simd
  do i = 1, len_1d
    a(i) = b(ip(i))
  end do
end subroutine tsvc_2_vag_fp64
