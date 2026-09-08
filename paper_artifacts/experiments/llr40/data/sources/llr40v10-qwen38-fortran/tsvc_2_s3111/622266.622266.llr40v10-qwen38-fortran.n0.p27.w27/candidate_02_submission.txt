subroutine tsvc_2_s3111_fp64(a, b, len_1d) bind(C, name='tsvc_2_s3111_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(in), dimension(*) :: a
  real(c_double), intent(inout), dimension(*) :: b
  integer(c_int64_t), intent(in), value :: len_1d
  integer(c_int64_t) :: i
  real(c_double) :: sum_val

  sum_val = 0.0d0
  !$omp parallel do reduction(+:sum_val) schedule(static)
  do i = 1, len_1d
    if (a(i) > 0.0d0) sum_val = sum_val + a(i)
  end do
  b(1) = sum_val
end subroutine
