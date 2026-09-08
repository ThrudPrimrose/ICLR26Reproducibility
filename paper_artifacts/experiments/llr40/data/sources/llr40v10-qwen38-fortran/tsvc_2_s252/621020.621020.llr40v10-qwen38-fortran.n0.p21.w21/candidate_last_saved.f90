subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_s252_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  integer(c_int64_t) :: i
  real(c_double) :: p, q

  if (len_1d <= 1) then
     if (len_1d == 1) a(1) = b(1) * c(1)
     return
  end if
  a(1) = b(1) * c(1)
  !$omp parallel do shared(a,b,c,len_1d) private(p,q) schedule(static)
  do i = 2, len_1d
     p = b(i) * c(i)
     q = b(i-1) * c(i-1)
     a(i) = p + q
  end do
end subroutine tsvc_2_s252_fp64
