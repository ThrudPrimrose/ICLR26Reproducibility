subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  real(c_double) :: a(*)
  real(c_double) :: b(*)
  integer(c_int32_t) :: ip(*)
  integer(c_int64_t), value :: LEN_1D
  real(c_double) :: tmp
      integer(c_int64_t) :: i
  !$omp parallel num_threads(1)
!$omp do
    do i = 1, LEN_1D
    tmp = b(ip(i) + 1)
        a(i) = tmp
  end do
    !$omp end do
    !$omp end parallel
end subroutine tsvc_2_vag_fp64
