subroutine tsvc_2_s1244_fp64(a, b, c, d, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s1244_fp64")
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   type(c_ptr), value, intent(in) :: workspace
   integer(c_int64_t), value, intent(in) :: workspace_size
   real(c_double), intent(inout) :: a(LEN_1D)
   real(c_double), intent(in) :: b(LEN_1D)
   real(c_double), intent(in) :: c(LEN_1D)
   real(c_double), intent(inout) :: d(LEN_1D)
   real(c_double), allocatable :: a_orig(:)
   integer(c_int64_t) :: i

   if (LEN_1D <= 1_c_int64_t) return

   allocate(a_orig(LEN_1D))
   a_orig = a

   !$omp parallel do simd default(none) shared(a,b,c,LEN_1D) private(i)
   do i = 1, LEN_1D-1
      a(i) = b(i) + c(i)*c(i) + b(i)*b(i) + c(i)
   end do

   !$omp parallel do simd default(none) shared(a, d, a_orig, LEN_1D) private(i)
   do i = 1, LEN_1D-1
      d(i) = a(i) + a_orig(i+1)
   end do

   deallocate(a_orig)
end subroutine tsvc_2_s1244_fp64
