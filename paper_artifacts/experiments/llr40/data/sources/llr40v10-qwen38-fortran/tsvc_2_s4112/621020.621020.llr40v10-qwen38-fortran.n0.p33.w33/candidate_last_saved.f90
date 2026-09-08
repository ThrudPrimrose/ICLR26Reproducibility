subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d) bind(C, name='tsvc_2_s4112_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout), dimension(len_1d) :: a
  real(c_double), intent(in),    dimension(len_1d) :: b
  integer(c_int32_t), intent(in), dimension(len_1d) :: ip
  integer(c_int64_t) :: i

  !$omp parallel do default(none) shared(a,b,ip,len_1d) private(i) schedule(static)
  do i = 1, len_1d
    a(i) = a(i) + b(ip(i)) * 2.0d0
  end do
end subroutine tsvc_2_s4112_fp64
