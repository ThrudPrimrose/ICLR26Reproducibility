subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name='tsvc_2_s152_fp64')
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), intent(in), value :: len_1d
  real(c_double), intent(inout), dimension(len_1d) :: a
  real(c_double), intent(inout), dimension(len_1d) :: b
  real(c_double), intent(in),    dimension(len_1d) :: c
  real(c_double), intent(in),    dimension(len_1d) :: d
  real(c_double), intent(in),    dimension(len_1d) :: e
  integer(c_int64_t) :: i

  !$omp parallel do default(none) shared(a,b,c,d,e,len_1d) schedule(static)
  do i = 1, len_1d
    b(i) = d(i) * e(i)
    a(i) = a(i) + b(i) * c(i)
  end do
end subroutine tsvc_2_s152_fp64
