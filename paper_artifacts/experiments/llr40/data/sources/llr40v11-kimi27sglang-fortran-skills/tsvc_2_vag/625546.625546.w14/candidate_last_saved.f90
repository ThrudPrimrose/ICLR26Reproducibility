subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C)
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in)    :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer :: i, n

  n = int(LEN_1D, kind=kind(i))
  !$omp parallel do simd simdlen(8) schedule(static) default(shared) private(i) nontemporal(a)
  do i = 1, n
    a(i) = b(ip(i))
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vag_fp64
