! Simple sequential prefix sum with C pointer arguments.
subroutine tsvc_2_s3112_fp64(a_ptr, b_ptr, n) bind(C, name='tsvc_2_s3112_fp64')
  use iso_c_binding, only: c_double, c_int64_t, c_ptr, c_f_pointer
  implicit none

  type(c_ptr), value, intent(in) :: a_ptr, b_ptr
  integer(c_int64_t), value, intent(in) :: n

  real(c_double), pointer :: a(:), b(:)
  integer(c_int64_t) :: i
  real(c_double) :: acc

  call c_f_pointer(a_ptr, a, [n])
  call c_f_pointer(b_ptr, b, [n])

  acc = 0.0_c_double
  do i = 1_c_int64_t, n
    acc = acc + a(i)
    b(i) = acc
  end do
end subroutine tsvc_2_s3112_fp64
