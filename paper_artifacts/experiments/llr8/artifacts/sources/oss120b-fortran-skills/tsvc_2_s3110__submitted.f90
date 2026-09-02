subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D, workspace, workspace_size) bind(C, name="tsvc_2_s3110_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   type(c_ptr), value, intent(in) :: workspace
   integer(c_int64_t), value, intent(in) :: workspace_size
   real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(inout) :: bb(2,2)
   integer(c_int64_t) :: i, j
   real(c_double) :: maxv
   integer(c_int64_t) :: xindex, yindex

   ! Initialize maxv to a very low value
   maxv = -huge(0.0d0)

   !$omp parallel do reduction(max:maxv) default(none) shared(aa, LEN_2D) private(i,j) schedule(static)
   do i = 1_c_int64_t, LEN_2D
      do j = 1_c_int64_t, LEN_2D
         if (aa(j,i) > maxv) maxv = aa(j,i)
      end do
   end do
   !$omp end parallel do

   ! Find the first occurrence of the maximum in row‑major order (Python’s order)
   xindex = -1_c_int64_t
   yindex = -1_c_int64_t
   do i = 1_c_int64_t, LEN_2D
      do j = 1_c_int64_t, LEN_2D
         if (aa(j,i) == maxv) then
            xindex = i - 1_c_int64_t
            yindex = j - 1_c_int64_t
            exit
         end if
      end do
      if (xindex >= 0_c_int64_t) exit
   end do

   ! Compute checksum and store in bb(1,1) (C‑order bb[0,0])
   bb(1,1) = maxv + dble(xindex) + dble(yindex)

end subroutine tsvc_2_s3110_fp64
