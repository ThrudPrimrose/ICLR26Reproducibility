subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(in) :: bb(LEN_2D, LEN_2D)
   integer(c_int64_t) :: i, j
   real(c_double) :: acc

   !$omp parallel private(i, j, acc)
   !$omp do schedule(static)
   do i = 1, LEN_2D
      acc = aa(i, 1)
      !$omp simd reduction(inscan, +: acc)
      do j = 2, LEN_2D
         acc = acc + bb(i, j)
         !$omp scan inclusive(acc)
         aa(i, j) = acc
      end do
   end do
   !$omp end do
   !$omp end parallel
end subroutine tsvc_2_s231_fp64
