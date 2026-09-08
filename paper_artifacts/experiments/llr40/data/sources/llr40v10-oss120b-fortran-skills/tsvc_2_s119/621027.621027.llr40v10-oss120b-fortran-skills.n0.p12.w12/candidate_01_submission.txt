subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
   integer(c_int64_t) :: i, j, t, i_start, i_end

   !$omp parallel default(none) shared(aa, bb, LEN_2D) private(t, i, j, i_start, i_end)
   do t = 4, 2*LEN_2D
      i_start = max(2_c_int64_t, t - LEN_2D)
      i_end   = min(LEN_2D, t - 2_c_int64_t)
      !$omp do
      do i = i_start, i_end
         j = t - i
         aa(j, i) = aa(j-1, i-1) + bb(j, i)
      end do
      !$omp end do
   end do
   !$omp end parallel

end subroutine tsvc_2_s119_fp64
