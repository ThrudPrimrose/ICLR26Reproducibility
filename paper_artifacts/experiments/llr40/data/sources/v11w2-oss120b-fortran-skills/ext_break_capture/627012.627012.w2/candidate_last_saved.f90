subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C)
   use iso_c_binding
   implicit none
   integer(c_int64_t), value :: LEN_1D
         real(c_double), intent(in) :: a(*)
   integer(c_int64_t), intent(inout) :: out_index(1)
   real(c_double), intent(inout) :: out_value(1)
   integer(c_int64_t) :: i
   real(c_double), parameter :: k = 1.0_c_double

   out_index(1) = -1_c_int64_t
   out_value(1) = -1.0_c_double

   do i = 0_c_int64_t, LEN_1D - 1_c_int64_t
      if (a(i+1) > k) then
         out_index(1) = i
         out_value(1) = a(i+1)
         exit
      end if
   end do

end subroutine ext_break_capture_fp64
