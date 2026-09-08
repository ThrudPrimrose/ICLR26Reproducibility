subroutine tsvc_2_s119_fp64(aa, bb, len_2d) bind(c, name="tsvc_2_s119_fp64")
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), intent(in), value :: aa
  type(c_ptr), intent(in), value :: bb
  integer(c_int64_t), intent(in), value :: len_2d
  real(c_double), dimension(:,:), pointer :: f, g
  integer :: n, i
  n = int(len_2d)
  call c_f_pointer(aa, f, [n, n])
  call c_f_pointer(bb, g, [n, n])
  do i = 2, n
     f(2:n, i) = f(1:n-1, i-1) + g(2:n, i)
  end do
end subroutine tsvc_2_s119_fp64
