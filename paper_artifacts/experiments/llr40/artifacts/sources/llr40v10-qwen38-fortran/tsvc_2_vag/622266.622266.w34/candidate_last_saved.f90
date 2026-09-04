subroutine tsvc_2_vag_fp64(a, b, ip, len1d) bind(C, name="tsvc_2_vag_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), value :: len1d
  real(c_double), intent(out) :: a(len1d)
  real(c_double), intent(in) :: b(len1d)
  integer(c_int32_t), intent(in) :: ip(len1d)

  integer(c_int64_t) :: i

  !$omp parallel do default(none) shared(a,b,ip,len1d) private(i)
  do i = 1, len1d
    a(i) = b(ip(i))
  end do
  !$omp end parallel do
end subroutine tsvc_2_vag_fp64
