subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   real(c_double), intent(inout) :: a(LEN_1D)
   real(c_double), intent(in) :: b(LEN_1D)
   real(c_double), intent(in) :: c(LEN_1D)
   real(c_double), intent(out) :: d(LEN_1D)
   real(c_double), allocatable :: a_orig(:)
   integer(c_int64_t) :: i

   if (LEN_1D > 0) then
      allocate(a_orig(LEN_1D))
      a_orig = a
   end if

   if (LEN_1D > 1) then
      !$omp parallel do default(none) shared(a, b, c, d, a_orig, LEN_1D) private(i)
      do i = 1, LEN_1D-1
         a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
         d(i) = a(i) + a_orig(i+1)
      end do
      !$omp end parallel do
   end if

   if (allocated(a_orig)) then
      deallocate(a_orig)
   end if
end subroutine tsvc_2_s1244_fp64
