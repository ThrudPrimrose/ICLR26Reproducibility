subroutine tsvc_2_s3111_fp64(a, b, len1d) bind(C, name="tsvc_2_s3111_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double) :: a(*)
  real(c_double) :: b(*)
  integer(c_int64_t), value :: len1d

  integer(c_int64_t) :: i
  real(c_double) :: s

  s = 0.0d0
  !$omp parallel do reduction(+:s) schedule(static)
  do i = 1, len1d
    if (a(i) > 0.0d0) s = s + a(i)
  end do
  !$omp end parallel do
  b(1) = s
end subroutine tsvc_2_s3111_fp64
