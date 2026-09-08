subroutine scan_affine_decay_fp64(c, x, y, LEN_1D) bind(C, name="scan_affine_decay_fp64")
   use iso_c_binding
   implicit none
   real(c_double), intent(in) :: c(*)
   real(c_double), intent(in) :: x(*)
   real(c_double), intent(out) :: y(*)
   integer(c_int64_t), value :: LEN_1D
   integer(c_int64_t) :: i

   if (LEN_1D <= 0_c_int64_t) then
      return
   end if

      ! Seed: set the first element from x
   y(1) = x(1)
   if (LEN_1D >= 2_c_int64_t) then
      do i = 2_c_int64_t, LEN_1D
         y(i) = c(i) * y(i-1) + x(i)
      end do
   end if

end subroutine scan_affine_decay_fp64