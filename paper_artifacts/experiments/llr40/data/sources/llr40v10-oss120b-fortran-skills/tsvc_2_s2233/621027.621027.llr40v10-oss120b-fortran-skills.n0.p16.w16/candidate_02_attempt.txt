subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_2D
   real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
   real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
   real(c_double), intent(in)    :: cc(LEN_2D, LEN_2D)
   integer(c_int64_t) :: i, j

   !$omp parallel default(none) shared(aa, bb, cc, LEN_2D) private(i, j)
   ! First recurrence (C: aa[j,i] = aa[j-1,i] + cc[j,i])
   !$omp do nowait
   do i = 9, LEN_2D          ! C i = 8 .. LEN_2D-1
      do j = 9, LEN_2D       ! C j = 8 .. LEN_2D-1
         aa(i, j) = aa(i, j-1) + cc(i, j)
      end do
   end do
   !$omp end do
   ! Second recurrence (C: bb[i,j] = bb[i-1,j] + cc[i,j])
   !$omp do
   do j = 9, LEN_2D          ! C j = 8 .. LEN_2D-1
      do i = 9, LEN_2D       ! C i = 8 .. LEN_2D-1
         bb(j, i) = bb(j, i-1) + cc(j, i)
      end do
   end do
   !$omp end do
   !$omp end parallel

end subroutine tsvc_2_s2233_fp64
