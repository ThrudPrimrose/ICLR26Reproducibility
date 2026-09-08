subroutine tsvc_2_s319_fp64(a, b, c, d, e, LEN_1D) bind(C)
   use iso_c_binding
   implicit none
   integer(c_int64_t), value :: LEN_1D
   real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D)
   real(c_double), intent(in) :: c(LEN_1D), d(LEN_1D), e(LEN_1D)
   integer(c_int64_t) :: i
   real(c_double) :: sum, tmp

      sum = 0.0_c_double
   !$omp parallel do simd reduction(+:sum) default(none) shared(a,b,c,d,e,LEN_1D) private(i, tmp)
   do i = 1, LEN_1D
      tmp = c(i)
      a(i) = tmp + d(i)
      b(i) = tmp + e(i)
      sum = sum + a(i) + b(i)
   end do
   b(1) = sum
end subroutine tsvc_2_s319_fp64
