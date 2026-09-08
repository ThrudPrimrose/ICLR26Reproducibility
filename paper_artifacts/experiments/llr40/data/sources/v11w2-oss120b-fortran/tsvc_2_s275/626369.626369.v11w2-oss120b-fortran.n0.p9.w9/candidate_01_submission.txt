subroutine tsvc_2_s275_fp64(aa, bb, cc, LEN_2D) bind(C, name="tsvc_2_s275_fp64")
   use iso_c_binding, only: c_double, c_int64_t
   implicit none
   integer(c_int64_t), value :: LEN_2D
   real(c_double), intent(inout) :: aa(*)
   real(c_double), intent(in) :: bb(*), cc(*)
   integer(c_int64_t) :: i, j, idx
   real(c_double) :: acc
   !$omp parallel do default(none) shared(aa, bb, cc, LEN_2D) private(i, j, idx, acc)
   do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
      if (aa(i+1) > 0.0_c_double) then
         ! Use accumulator to avoid repeated load of aa
         acc = aa(i+1)
         do j = 1_c_int64_t, LEN_2D - 1_c_int64_t
            idx = j*LEN_2D + i + 1
            acc = acc + bb(idx) * cc(idx)
            aa(idx) = acc
         end do
      end if
   end do
   !$omp end parallel do
end subroutine tsvc_2_s275_fp64
