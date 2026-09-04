subroutine tsvc_2_s119_fp64(aa, bb, len_2d) bind(C, name='tsvc_2_s119_fp64')
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: aa
  type(c_ptr), value, intent(in) :: bb
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), contiguous, pointer :: a(:,:), b(:,:)
  integer :: n, i
  n = int(len_2d, kind=4)
  call c_f_pointer(aa, a, [n, n])
  call c_f_pointer(bb, b, [n, n])
  do i = 2, n
    a(2:n, i) = a(1:n-1, i-1) + b(2:n, i)
  end do
end subroutine tsvc_2_s119_fp64
