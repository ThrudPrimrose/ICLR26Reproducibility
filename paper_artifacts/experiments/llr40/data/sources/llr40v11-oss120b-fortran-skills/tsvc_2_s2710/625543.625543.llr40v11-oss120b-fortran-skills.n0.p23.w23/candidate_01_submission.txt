subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, LEN_1D, workspace, workspace_size) bind(C)
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D
   type(c_ptr), value, intent(in) :: workspace
   integer(c_int64_t), value, intent(in) :: workspace_size
   real(c_double), intent(inout) :: a(LEN_1D), b(LEN_1D), c(LEN_1D)
   real(c_double), intent(in) :: d(LEN_1D), e(LEN_1D), x(LEN_1D)
   integer(c_int64_t) :: i
   logical :: len_big, x_pos

   len_big = LEN_1D > 10
   x_pos = x(1) > 0.0d0

   if (len_big) then
      if (x_pos) then
         !$omp simd
         do i = 1, LEN_1D
            if (a(i) > b(i)) then
               a(i) = a(i) + b(i) * d(i)
               c(i) = c(i) + d(i) * d(i)
            else
               b(i) = a(i) + e(i) * e(i)
               c(i) = a(i) + d(i) * d(i)
            end if
         end do
      else
         !$omp simd
         do i = 1, LEN_1D
            if (a(i) > b(i)) then
               a(i) = a(i) + b(i) * d(i)
               c(i) = c(i) + d(i) * d(i)
            else
               b(i) = a(i) + e(i) * e(i)
               c(i) = c(i) + e(i) * e(i)
            end if
         end do
      end if
   else
      if (x_pos) then
         !$omp simd
         do i = 1, LEN_1D
            if (a(i) > b(i)) then
               a(i) = a(i) + b(i) * d(i)
               c(i) = d(i) * e(i) + 1.0d0
            else
               b(i) = a(i) + e(i) * e(i)
               c(i) = a(i) + d(i) * d(i)
            end if
         end do
      else
         !$omp simd
         do i = 1, LEN_1D
            if (a(i) > b(i)) then
               a(i) = a(i) + b(i) * d(i)
               c(i) = d(i) * e(i) + 1.0d0
            else
               b(i) = a(i) + e(i) * e(i)
               c(i) = c(i) + e(i) * e(i)
            end if
         end do
      end if
   end if

end subroutine tsvc_2_s2710_fp64
