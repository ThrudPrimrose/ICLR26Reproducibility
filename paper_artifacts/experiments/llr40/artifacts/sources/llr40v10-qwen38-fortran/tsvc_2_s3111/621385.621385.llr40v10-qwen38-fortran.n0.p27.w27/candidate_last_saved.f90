! TSVC tsvc_2/s3111: b(0) = sum of a(i) over i with a(i) > 0.
! Parallel reduction; per-thread chunk auto-vectorizes (masked add).
subroutine tsvc_2_s3111_fp64(a, b, len_1d) bind(C, name='tsvc_2_s3111_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(kind=c_double), intent(in)  :: a(*)
  real(kind=c_double), intent(out) :: b(*)
  integer(kind=c_int64_t), value  :: len_1d
  real(kind=c_double) :: sum
  integer(kind=c_int64_t) :: i

  sum = 0.0d0
  !$omp parallel do reduction(+:sum) schedule(static)
  do i = 1, len_1d
    if (a(i) > 0.0d0) sum = sum + a(i)
  end do
  b(1) = sum
end subroutine
