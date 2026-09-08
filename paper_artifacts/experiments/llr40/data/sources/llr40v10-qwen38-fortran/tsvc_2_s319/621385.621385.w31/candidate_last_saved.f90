subroutine tsvc_2_s319_fp64(a, b, c, d, e, len_1d) bind(C, name='tsvc_2_s319_fp64')
   use, intrinsic :: iso_c_binding
   implicit none
   real(c_double), intent(inout) :: a(*)
   real(c_double), intent(inout) :: b(*)
   real(c_double), intent(in)    :: c(*)
   real(c_double), intent(in)    :: d(*)
   real(c_double), intent(in)    :: e(*)
   integer(c_int64_t), value :: len_1d
   integer(c_int64_t) :: i
   real(c_double) :: sum

   sum = 0.0d0
   do i = 0, len_1d - 1
      a(i+1) = c(i+1) + d(i+1)
      sum = sum + a(i+1)
      b(i+1) = c(i+1) + e(i+1)
      sum = sum + b(i+1)
   end do
   b(1) = sum
end subroutine tsvc_2_s319_fp64
