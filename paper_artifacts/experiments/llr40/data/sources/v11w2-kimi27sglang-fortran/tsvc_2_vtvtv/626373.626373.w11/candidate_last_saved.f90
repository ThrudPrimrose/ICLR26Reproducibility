! TSVC tsvc_2 kernel vtvtv - a(i) = a(i) * b(i) * c(i)
module tsvc_2_vtvtv_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_vtvtv_fp64(a, b, c, LEN_1D) bind(c, name="tsvc_2_vtvtv_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: c(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i

    !$omp parallel do simd safelen(8) simdlen(8) schedule(static) if(LEN_1D > 65536)
    do i = 1, LEN_1D
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_vtvtv_fp64
end module tsvc_2_vtvtv_mod
