! Kernel for tsvc_2_s119 (fp64 version) - diagonal processing with accumulator for minimal memory traffic
subroutine tsvc_2_s119_fp64(aa, bb, LEN_2D, workspace, workspace_bytes) bind(C, name="tsvc_2_s119_fp64")
   use iso_c_binding
   integer(c_int64_t), value :: LEN_2D
   type(c_ptr), value :: workspace
   integer(c_int64_t), value :: workspace_bytes
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
   integer :: d, i, i_start, i_end, j
   real(c_double) :: acc

   ! Parallelize over diagonals (i - j = d). Each diagonal can be computed independently.
   !$omp parallel do private(d, i, i_start, i_end, j, acc) schedule(static)
   do d = -(LEN_2D-2), LEN_2D-2
      i_start = max(2, d+2)
      i_end   = min(LEN_2D, LEN_2D + d)
      j = i_start - d
      ! Initialize accumulator with the element just before the start of this diagonal.
      acc = aa(i_start-1, j-1) + bb(i_start, j)
      aa(i_start, j) = acc
      do i = i_start+1, i_end
         j = i - d
         acc = acc + bb(i, j)
         aa(i, j) = acc
      end do
   end do
   !$omp end parallel do

   ! workspace arguments unused but required by harness
end subroutine tsvc_2_s119_fp64
