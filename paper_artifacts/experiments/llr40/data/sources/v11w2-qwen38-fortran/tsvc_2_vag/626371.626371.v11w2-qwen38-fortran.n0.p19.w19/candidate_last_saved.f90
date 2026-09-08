subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
  use, intrinsic :: iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  real(kind=c_double), intent(inout) :: a(*)
  real(kind=c_double), intent(in)    :: b(*)
  integer(kind=c_int32_t),  intent(in) :: ip(*)
  integer(kind=c_int64_t),  value      :: len_1d
  integer(kind=c_int64_t) :: i

  ! a[i] = b[ip[i]]  -- ip index buffer is delivered 1-based (Fortran base),
  ! so we subscript b directly with the value handed to us.
  !$omp parallel do default(none) shared(a,b,ip,len_1d) private(i)
  do i = 1, len_1d
    a(i) = b(ip(i))
  end do
  !$omp end parallel do
end subroutine tsvc_2_vag_fp64
