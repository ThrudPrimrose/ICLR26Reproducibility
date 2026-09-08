subroutine tsvc_2_s2275_fp64(a, aa, b, bb, c, cc, d, len_2d) bind(C, name="tsvc_2_s2275_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: len_2d
   real(c_double), intent(inout) :: a(len_2d)
   real(c_double), intent(inout) :: aa(len_2d, len_2d)
   real(c_double), intent(in) :: b(len_2d)
   real(c_double), intent(in) :: bb(len_2d, len_2d)
   real(c_double), intent(in) :: c(len_2d)
   real(c_double), intent(in) :: cc(len_2d, len_2d)
   real(c_double), intent(in) :: d(len_2d)
   integer(c_int64_t) :: i, j
   ! Parallel region for 2D update with SIMD inner loop
   !$omp parallel default(none) shared(a,aa,b,bb,c,cc,d,len_2d) private(i,j)
      !$omp do schedule(static)
      do j = 1, len_2d
         !$omp simd
         do i = 1, len_2d
            aa(i,j) = aa(i,j) + bb(i,j) * cc(i,j)
         end do
      end do
      !$omp end do
      ! Compute 1D vector a in parallel
      !$omp do schedule(static)
      do i = 1, len_2d
         a(i) = b(i) + c(i) * d(i)
      end do
      !$omp end do
   !$omp end parallel
end subroutine tsvc_2_s2275_fp64
