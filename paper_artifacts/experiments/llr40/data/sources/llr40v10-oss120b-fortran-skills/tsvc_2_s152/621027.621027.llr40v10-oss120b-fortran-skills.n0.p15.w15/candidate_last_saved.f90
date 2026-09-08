subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d) bind(C)
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: len_1d
   real(c_double), intent(inout) :: a(len_1d)
   real(c_double), intent(inout) :: b(len_1d)
   real(c_double), intent(in) :: c(len_1d)
   real(c_double), intent(in) :: d(len_1d)
   real(c_double), intent(in) :: e(len_1d)
   integer(c_int64_t) :: i
   real(c_double) :: temp

   !$omp parallel do simd default(none) shared(a,b,c,d,e,len_1d) private(i,temp)
   do i = 1, len_1d
      temp = d(i) * e(i)
      b(i) = temp
      a(i) = a(i) + temp * c(i)
   end do
   end subroutine tsvc_2_s152_fp64
