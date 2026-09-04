module tsvc_2_vtvtv_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vtvtv_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_vtvtv_fp64")
    implicit none
    real(c_double) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    !$omp parallel do simd default(none) schedule(static) shared(a,b,c,len_1d) private(i)
    do i = 1, len_1d
      a(i) = a(i) * b(i) * c(i)
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_vtvtv_fp64
end module tsvc_2_vtvtv_mod
