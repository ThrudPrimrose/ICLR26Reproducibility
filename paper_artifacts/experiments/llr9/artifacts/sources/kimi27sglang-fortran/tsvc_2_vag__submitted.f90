module tsvc_2_vag_mod
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(c, name="tsvc_2_vag_fp64")
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in)  :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), intent(in), value :: LEN_1D
    integer(c_int64_t) :: i, n4
    !$omp parallel do schedule(static) private(i)
    do i = 1, LEN_1D - 3, 4
      a(i)   = b(ip(i))
      a(i+1) = b(ip(i+1))
      a(i+2) = b(ip(i+2))
      a(i+3) = b(ip(i+3))
    end do
    !$omp end parallel do
    n4 = LEN_1D - mod(LEN_1D, 4_c_int64_t)
    do i = n4 + 1, LEN_1D
      a(i) = b(ip(i))
    end do
  end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
