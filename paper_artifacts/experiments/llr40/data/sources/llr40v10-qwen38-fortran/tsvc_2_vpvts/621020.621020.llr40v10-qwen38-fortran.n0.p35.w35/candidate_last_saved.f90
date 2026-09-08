      subroutine tsvc_2_vpvts_fp64(a, b, len_1d, s) bind(c, name="tsvc_2_vpvts_fp64")
      use iso_c_binding
      implicit none
      integer(c_int64_t), value     :: len_1d
      integer(c_int64_t), value     :: s
      real(c_double), intent(inout) :: a(len_1d)
      real(c_double), intent(in)    :: b(len_1d)
      integer :: i

!$omp parallel do
      do i = 1, int(len_1d)
         a(i) = a(i) + b(i) * real(s, c_double)
      end do
      end subroutine
