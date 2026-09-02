subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C, name="tsvc_2_s119_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   type(c_ptr), value, intent(in) :: workspace
   integer(c_int64_t), value, intent(in) :: workspace_size
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
   integer(c_int64_t) :: t, i, j, i_start, i_end

   !$omp parallel private(t,i,j,i_start,i_end)
   do t = 4_c_int64_t, 2_c_int64_t*LEN_2D
      i_start = max(2_c_int64_t, t - LEN_2D)
      i_end   = min(LEN_2D, t - 2_c_int64_t)
      !$omp do schedule(static)
      do i = i_start, i_end
         j = t - i
         aa(i,j) = aa(i-1,j-1) + bb(i,j)
      end do
      !$omp end do
   end do
   !$omp end parallel
end subroutine tsvc_2_s119_fp64
