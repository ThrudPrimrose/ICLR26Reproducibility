subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(c, name="tsvc_2_vpvts_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*)
  real(c_double), intent(in) :: b(*)
  integer(c_int64_t), value :: len_1d
  integer(c_int64_t), value :: s

  integer :: i
  real(c_double) :: sf
  sf = real(s, c_double)
  !$omp parallel do default(none) shared(a, b, len_1d, sf) private(i)
  do i = 1, int(len_1d)
    a(i) = a(i) + b(i) * sf
  end do
  !$omp end parallel do
end subroutine tsvc_2_vpvts_fp64
