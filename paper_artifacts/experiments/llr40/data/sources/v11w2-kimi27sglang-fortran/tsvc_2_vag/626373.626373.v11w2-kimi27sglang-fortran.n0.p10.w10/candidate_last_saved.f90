subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
  integer(c_int64_t), value :: LEN_1D
  real(c_double), intent(inout) :: a(LEN_1D)
  real(c_double), intent(in) :: b(LEN_1D)
  integer(c_int32_t), intent(in) :: ip(LEN_1D)
  integer(c_int64_t) :: i
  integer(c_int64_t) :: idx
  !$omp parallel do simd simdlen(16) schedule(static) private(idx) aligned(a:64,b:8,ip:4)
  do i = 1_c_int64_t, LEN_1D
    idx = int(ip(i), c_int64_t)
    a(i) = b(idx)
  end do
  !$omp end parallel do simd
end subroutine tsvc_2_vag_fp64
