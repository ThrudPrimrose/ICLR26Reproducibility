module tsvc_2_s152_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C, name="tsvc_2_s152_fp64")
    real(c_double), intent(inout) :: a(0:*)
    real(c_double), intent(inout) :: b(0:*)
    real(c_double), intent(in) :: c(0:*), d(0:*), e(0:*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    !$omp parallel do simd default(none) private(i) shared(a,b,c,d,e,len_1d)
    do i = 0, len_1d - 1
      b(i) = d(i) * e(i)
      a(i) = a(i) + b(i) * c(i)
    end do
    !$omp end parallel do simd
  end subroutine tsvc_2_s152_fp64
end module tsvc_2_s152_mod
