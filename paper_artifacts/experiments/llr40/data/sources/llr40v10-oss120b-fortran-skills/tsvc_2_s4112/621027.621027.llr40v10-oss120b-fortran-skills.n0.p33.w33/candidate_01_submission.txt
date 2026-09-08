subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   real(c_double), intent(inout) :: a(LEN_1D)
   real(c_double), intent(in) :: b(LEN_1D)
   integer(c_int32_t), intent(in) :: ip(LEN_1D)
   integer(c_int64_t) :: i
   real(c_double) :: ai, bi

   !$omp parallel do default(none) shared(a, b, ip, LEN_1D) private(i, ai, bi)
   do i = 1, LEN_1D
      ai = a(i)
      bi = b(ip(i))
      a(i) = ai + bi * 2.0d0
   end do
   !$omp end parallel do
end subroutine tsvc_2_s4112_fp64
