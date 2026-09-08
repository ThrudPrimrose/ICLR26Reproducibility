subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
   integer(c_int64_t) :: i, j
   logical(c_bool) :: active(LEN_2D)

   ! Determine which rows need processing based on first column
   do i = 1, LEN_2D
      active(i) = aa(i, 1) > 0.0d0
   end do

   do j = 2, LEN_2D
      !$omp simd
      do i = 1, LEN_2D
         if (active(i)) then
            aa(i, j) = aa(i, j-1) + bb(i, j) * cc(i, j)
         end if
      end do
      !$omp end simd
   end do
end subroutine tsvc_2_s275_fp64
