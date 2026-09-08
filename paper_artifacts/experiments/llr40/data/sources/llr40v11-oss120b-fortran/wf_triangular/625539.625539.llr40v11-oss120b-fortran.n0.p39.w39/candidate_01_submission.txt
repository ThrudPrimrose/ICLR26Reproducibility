subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
   use iso_c_binding
   implicit none
   real(c_double), intent(inout) :: a(*)
   integer(c_int64_t), value :: LEN_2D
   integer(c_int64_t) :: i, j, s, start_i, end_i

   !$omp parallel private(i, j, s, start_i, end_i)
   do s = 2_c_int64_t, 2_c_int64_t*LEN_2D - 2_c_int64_t
      start_i = max(1_c_int64_t, s - (LEN_2D - 1_c_int64_t))
      end_i = min(LEN_2D - 1_c_int64_t, s / 2_c_int64_t)
      !$omp do schedule(static) nowait
      do i = start_i, end_i
         j = s - i
         a(i*LEN_2D + j + 1) = a(i*LEN_2D + j + 1) + a((i-1)*LEN_2D + j + 1) + a(i*LEN_2D + (j-1) + 1)
      end do
      !$omp end do
      !$omp barrier
   end do
   !$omp end parallel
end subroutine wf_triangular_fp64
