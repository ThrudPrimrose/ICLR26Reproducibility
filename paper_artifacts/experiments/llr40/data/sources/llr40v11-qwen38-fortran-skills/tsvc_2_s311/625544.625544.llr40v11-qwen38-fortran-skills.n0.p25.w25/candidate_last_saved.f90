subroutine tsvc_2_s311_fp64(a, sum_out, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: sum_out(*)

  real(c_double) :: s
  integer(kind=c_int64_t) :: i, n

  n = len_1d
  s = 0.0d0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + a(i)
  end do
  sum_out(1) = s
end subroutine tsvc_2_s311_fp64
