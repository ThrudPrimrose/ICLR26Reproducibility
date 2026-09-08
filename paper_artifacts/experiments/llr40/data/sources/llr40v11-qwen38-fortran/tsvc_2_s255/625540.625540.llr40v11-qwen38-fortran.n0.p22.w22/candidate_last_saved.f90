subroutine tsvc_2_s255_fp64(a, b, LEN_1D, ws, ws_size) bind(C, name="tsvc_2_s255_fp64")
   use, intrinsic :: iso_c_binding
   implicit none
   real(c_double), dimension(*), intent(out) :: a
   real(c_double), dimension(*), intent(in) :: b
   integer(c_int64_t), value, intent(in) :: LEN_1D
   type(c_ptr), value :: ws
   integer(c_int64_t), value, intent(in) :: ws_size
   integer(c_int64_t) :: n, i

   n = LEN_1D
   if (n < 3) then
      if (n == 1) then
         a(1) = (b(1) + b(1) + b(1)) * 0.333d0
         return
      end if
      if (n == 2) then
         a(1) = (b(1) + b(2) + b(1)) * 0.333d0
         a(2) = (b(2) + b(1) + b(2)) * 0.333d0
         return
      end if
      return
   end if

   a(1) = (b(1) + b(n) + b(n-1)) * 0.333d0
   a(2) = (b(2) + b(1) + b(n)) * 0.333d0

   !$omp parallel do schedule(static) private(i)
   do i = 3, n
      a(i) = (b(i) + b(i-1) + b(i-2)) * 0.333d0
   end do
end subroutine tsvc_2_s255_fp64
