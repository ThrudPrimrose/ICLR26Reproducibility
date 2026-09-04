subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(c, name='quasi_affine_reduce_odd_fp64')
  use iso_c_binding, only: c_ptr, c_double, c_int64_t, c_f_pointer
  implicit none
  type(c_ptr), intent(in) :: a, out
  integer(c_int64_t), intent(in) :: len_1d
  real(c_double), dimension(:), pointer :: fa
  real(c_double), dimension(:), pointer :: fo
  real(c_double) :: acc
  integer(c_int64_t) :: i

  call c_f_pointer(a, fa, [len_1d])
  call c_f_pointer(out, fo, [1])

  acc = 0.0d0
  do i = 2, len_1d - 1, 2
    acc = acc + fa(i)
  end do
  fo(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
