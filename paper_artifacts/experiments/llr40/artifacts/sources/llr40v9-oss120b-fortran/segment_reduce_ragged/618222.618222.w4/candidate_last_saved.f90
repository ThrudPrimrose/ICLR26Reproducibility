subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, workspace, workspace_bytes) bind(C, name="segment_reduce_ragged_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   ! Arguments
   real(c_double), intent(out) :: out(*)
   integer(c_int64_t), intent(in) :: row_ptr(*)
   real(c_double), intent(in) :: val(*)
   real(c_double), intent(in) :: w(*)
   integer(c_int64_t), value :: NSEG
   type(c_ptr), value :: workspace   ! unused
   integer(c_int64_t), value :: workspace_bytes   ! unused
   ! Local variables
   integer(c_int64_t) :: s, e, start_idx, end_idx
   real(c_double) :: acc

   if (NSEG <= 0_c_int64_t) return

   !$omp parallel do schedule(guided) private(s, start_idx, end_idx, e, acc)
   do s = 1_c_int64_t, NSEG
      start_idx = row_ptr(s) + 1_c_int64_t
      end_idx = row_ptr(s+1)
      acc = 0.0_c_double
      !$omp simd reduction(+:acc)
      do e = start_idx, end_idx
         acc = acc + val(e) * w(e)
      end do
      out(s) = acc
   end do
   !$omp end parallel do

end subroutine segment_reduce_ragged_fp64
