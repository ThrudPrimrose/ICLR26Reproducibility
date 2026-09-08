! TSVC tsvc_2 vtvtv: a(i) = a(i) * b(i) * c(i)
! Fortran implementation with OpenMP parallelism and vectorization.
subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_vtvtv_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in)    :: b(len_1d)
  real(c_double), intent(in)    :: c(len_1d)

  integer(c_int64_t) :: i

  if (len_1d <= 0) return

  !$omp parallel do default(none) shared(a, b, c, len_1d) schedule(static)
  do i = 1_8, len_1d
    a(i) = a(i) * b(i) * c(i)
  end do
end subroutine tsvc_2_vtvtv_fp64
