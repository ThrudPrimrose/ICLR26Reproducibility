subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C, name="tsvc_2_s1244_fp64")
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   real(c_double), intent(inout) :: a(*)
   real(c_double), intent(in) :: b(*)
   real(c_double), intent(in) :: c(*)
   real(c_double), intent(inout) :: d(*)
   integer(c_int64_t) :: i
real(c_double), allocatable :: a_old(:)
real(c_double) :: new_a

   allocate(a_old(LEN_1D))
   a_old(1:LEN_1D) = a(1:LEN_1D)
   !$omp parallel do simd schedule(static) private(new_a)
   do i = 1, LEN_1D - 1
     new_a = b(i) + c(i) * c(i) + b(i) * b(i) + c(i)
     a(i) = new_a
     d(i) = new_a + a_old(i + 1)
   end do
   deallocate(a_old)

end subroutine tsvc_2_s1244_fp64
