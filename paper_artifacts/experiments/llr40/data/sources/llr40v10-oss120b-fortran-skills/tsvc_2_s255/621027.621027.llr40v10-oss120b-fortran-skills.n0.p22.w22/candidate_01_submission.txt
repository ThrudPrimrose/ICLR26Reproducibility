subroutine tsvc_2_s255_fp64(a, b, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  integer(c_int64_t) :: i, i1, i2
  !$omp parallel do default(none) shared(a,b,len_1d) private(i,i1,i2) schedule(static)
  do i = 1, len_1d
    i1 = i - 1
   if (i1 < 1) i1 = len_1d
   i2 = i - 2
   if (i2 < 1) i2 = i2 + len_1d
   a(i) = (b(i) + b(i1) + b(i2)) * 0.333d0
  end do
  !$omp end parallel do
end subroutine tsvc_2_s255_fp64
