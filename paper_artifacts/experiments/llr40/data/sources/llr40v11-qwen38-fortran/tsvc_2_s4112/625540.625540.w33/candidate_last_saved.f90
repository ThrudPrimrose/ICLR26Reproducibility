module tsvc_mod
contains
subroutine worker(a, b, ip, n)
  use, intrinsic :: iso_c_binding
  implicit none
  real(kind=8), intent(inout) :: a(:)
  real(kind=8), intent(in) :: b(:)
  integer(c_int32_t), intent(in) :: ip(:)
  integer, intent(in) :: n
  integer :: i

  !$omp parallel do default(none) shared(a, b, ip, n) schedule(static)
  do i = 1, n
    a(i) = a(i) + 2.0d0 * b(ip(i))
  end do
  !$omp end parallel do
end subroutine worker
end module tsvc_mod

subroutine tsvc_2_s4112_fp64(a, b, ip, len_1d) bind(C, name='tsvc_2_s4112_fp64')
  use tsvc_mod
  use, intrinsic :: iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: a
  type(c_ptr), value, intent(in) :: b
  type(c_ptr), value, intent(in) :: ip
  integer(c_int64_t), value, intent(in) :: len_1d

  real(kind=8), pointer, contiguous :: fa(:), fb(:)
  integer(c_int32_t), pointer, contiguous :: fip(:)
  integer(c_int64_t) :: n

  n = len_1d
  call c_f_pointer(a, fa, [n])
  call c_f_pointer(b, fb, [n])
  call c_f_pointer(ip, fip, [n])
  call worker(fa, fb, fip, int(n))
end subroutine tsvc_2_s4112_fp64
