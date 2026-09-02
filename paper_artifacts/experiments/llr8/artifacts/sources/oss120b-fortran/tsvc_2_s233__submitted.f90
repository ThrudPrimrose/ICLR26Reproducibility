subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D, workspace, workspace_bytes) bind(C, name="tsvc_2_s233_fp64")
   use iso_c_binding
   implicit none
   integer(c_int64_t), value :: LEN_2D
   type(c_ptr), value :: workspace
   integer(c_int64_t), value :: workspace_bytes
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
   real(c_double), intent(in)    :: cc(LEN_2D, LEN_2D)
   integer :: i, j

   ! Compute aa vertical prefix sum for columns >=9 (0-index 8)
   !$omp parallel do schedule(static) private(j) default(none) shared(aa, cc, LEN_2D)
   do i = 9, LEN_2D
      do j = 9, LEN_2D
         aa(i,j) = aa(i,j-1) + cc(i,j)
      end do
   end do
   !$omp end parallel do

   ! Compute bb using previous column values; outer loop must be sequential due to dependence.
   do i = 9, LEN_2D
      !$omp simd
      do j = 9, LEN_2D
         bb(i,j) = bb(i-1,j) + cc(i,j)
      end do
   end do

end subroutine tsvc_2_s233_fp64
