subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name="tsvc_2_s235_fp64")
   use iso_c_binding, only: c_double, c_int64_t, c_int
   implicit none
   real(c_double), intent(inout) :: a(*)
   real(c_double), intent(inout) :: aa(*)
   real(c_double), intent(in) :: b(*)
   real(c_double), intent(in) :: bb(*)
   real(c_double), intent(in) :: c(*)
   integer(c_int64_t), value :: LEN_2D
   integer(c_int) :: i, j
   integer(c_int64_t) :: idx
   real(c_double) :: prev, ai

   !$omp parallel do private(i, j, idx, prev, ai)
   do i = 1, LEN_2D
      a(i) = a(i) + b(i) * c(i)
      ai = a(i)
      prev = aa(i)
      idx = i + LEN_2D
      do j = 2, LEN_2D
         prev = prev + bb(idx) * ai
         aa(idx) = prev
         idx = idx + LEN_2D
      end do
   end do
   !$omp end parallel do

end subroutine tsvc_2_s235_fp64
