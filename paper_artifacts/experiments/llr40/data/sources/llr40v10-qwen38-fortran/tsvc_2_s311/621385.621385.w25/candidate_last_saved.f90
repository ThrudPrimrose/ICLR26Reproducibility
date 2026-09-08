subroutine tsvc_2_s311_fp64(a, sum_out, len_1d) bind(C, name="tsvc_2_s311_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(kind=c_double), intent(in) :: a(*)
  real(kind=c_double), intent(out) :: sum_out(*)
  integer(kind=c_int64_t), value, intent(in) :: len_1d
  integer(kind=c_int64_t) :: i
  real(kind=c_double) :: s

  s = 0.0d0
  !$omp parallel do reduction(+:s) schedule(static)
  do i = 1, len_1d
    s = s + a(i)
  end do
  !$omp end parallel do
  sum_out(1) = s
end subroutine tsvc_2_s311_fp64
