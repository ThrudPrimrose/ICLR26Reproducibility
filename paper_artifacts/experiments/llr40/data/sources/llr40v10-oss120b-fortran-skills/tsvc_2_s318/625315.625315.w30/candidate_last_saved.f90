subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C)
   use iso_c_binding
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D, inc
   ! The array a must be large enough to hold all accessed elements:
   ! indices 0, inc, 2*inc, ..., (LEN_1D-1)*inc which correspond to Fortran indices 1, inc+1, ...
   real(c_double), intent(in) :: a(((LEN_1D - 1_c_int64_t) * inc + 1_c_int64_t))
   real(c_double), intent(out) :: result(1)

   integer(c_int64_t) :: i
   real(c_double) :: v, maxv
   integer(c_int64_t) :: index

   ! First reduction: find maximum absolute value.
   maxv = -huge(0.0_c_double)
   !$omp parallel do reduction(max:maxv) schedule(static)
   do i = 1, LEN_1D
      v = abs(a((i - 1_c_int64_t) * inc + 1_c_int64_t))
      if (v > maxv) maxv = v
   end do
   !$omp end parallel do

   ! Second reduction: find the first index where the maximum occurs.
   index = LEN_1D   ! sentinel larger than any valid index
   !$omp parallel do reduction(min:index) schedule(static)
   do i = 1, LEN_1D
      v = abs(a((i - 1_c_int64_t) * inc + 1_c_int64_t))
      if (v == maxv) then
         index = min(index, i - 1_c_int64_t)
      end if
   end do
   !$omp end parallel do

   result(1) = maxv + dble(index)
end subroutine tsvc_2_s318_fp64