subroutine versioned_distance_update_fp64(a, b, c, LEN_1D, K) bind(C)
   use iso_c_binding
   use omp_lib
   implicit none
   integer(c_int64_t), value, intent(in) :: LEN_1D, K
   real(c_double), intent(inout) :: a(LEN_1D)
   real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
   integer(c_int64_t) :: i
   integer(c_int) :: tid, nth, start

   if (K <= 1_c_int64_t) then
      do i = K + 1_c_int64_t, LEN_1D
         a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
      end do
   else
      !$omp parallel private(tid, nth, start, i) default(none) &
      !$omp& shared(a, b, c, LEN_1D, K)
      tid = omp_get_thread_num()
      nth = omp_get_num_threads()
      start = tid + 1
      if (start <= K) then
         do i = start + K, LEN_1D, K
            a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
         end do
      end if
      !$omp end parallel
   end if

end subroutine versioned_distance_update_fp64
