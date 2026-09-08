subroutine scatter_accum_dup_fp64(bins, ip, src, LEN_1D) bind(C, name="scatter_accum_dup_fp64")
   use iso_c_binding
   use omp_lib
   implicit none
   real(c_double), intent(inout) :: bins(*)
   real(c_double), intent(in) :: src(*)
   integer(c_int), intent(in) :: ip(*)
   integer(c_int64_t), value :: LEN_1D
   integer(c_int64_t) :: i

   if (LEN_1D <= 0_c_int64_t) then
      return
   end if

   !$omp parallel do schedule(static) private(i)
   do i = 1_c_int64_t, LEN_1D
      ! ip values are zero-based; convert to Fortran 1-based index
      !$omp atomic
      bins(ip(i)) = bins(ip(i)) + src(i)
   end do
   !$omp end parallel do

end subroutine scatter_accum_dup_fp64
