module tsvc_2_vag_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i

    !$omp parallel do simd nontemporal(a)
    do i = 1, LEN_1D
      a(i) = b(ip(i))
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
