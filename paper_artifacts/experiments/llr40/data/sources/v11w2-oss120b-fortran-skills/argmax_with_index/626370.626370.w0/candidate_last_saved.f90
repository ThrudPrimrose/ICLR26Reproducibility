subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   real(c_double), intent(in) :: a(*)
   integer(c_int64_t), intent(out) :: out_index
   real(c_double), intent(out) :: out_value
   integer(c_int64_t) :: i
   real(c_double), volatile :: max_val
   integer(c_int64_t) :: idx

   if (LEN_1D <= 0_c_int64_t) then
      out_index = 0_c_int64_t
      out_value = 0.0_c_double
      return
   end if

   max_val = a(1)
   idx = 0_c_int64_t
   i = 2_c_int64_t
   do while (i <= LEN_1D)
      if (a(i) > max_val) then
         max_val = a(i)
         idx = i - 1_c_int64_t
      end if
      i = i + 1_c_int64_t
   end do

   out_value = max_val
   out_index = idx
end subroutine argmax_with_index_fp64
