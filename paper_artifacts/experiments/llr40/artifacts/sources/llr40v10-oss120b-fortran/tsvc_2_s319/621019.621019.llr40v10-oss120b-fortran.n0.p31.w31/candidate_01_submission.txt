module tsvc_2_s319_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C, name="tsvc_2_s319_fp64")
    real(C_DOUBLE), intent(inout) :: a(0:*)
    real(C_DOUBLE), intent(out) :: b(0:*)
    real(C_DOUBLE), intent(in) :: c(0:*), d(0:*), e(0:*)
    integer(C_INT64_T), value :: LEN_1D
    integer(C_INT64_T) :: i
    real(C_DOUBLE) :: sum
    real(C_DOUBLE) :: ci
    sum = 0.0_C_DOUBLE
    !$omp parallel do default(none) shared(a,b,c,d,e,LEN_1D) private(i, ci) reduction(+:sum)
    do i = 0, LEN_1D - 1
      ci = c(i)
      a(i) = ci + d(i)
      sum = sum + a(i)
      b(i) = ci + e(i)
      sum = sum + b(i)
    end do
    !$omp end parallel do
    b(0) = sum
  end subroutine tsvc_2_s319_fp64
end module tsvc_2_s319_mod
