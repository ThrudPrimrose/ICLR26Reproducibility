subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(C, name='quasi_affine_reduce_odd_fp64')
  use iso_c_binding
  implicit none
  real(c_double), intent(in)    :: a(*)
  real(c_double), intent(out)   :: out(*)
  integer(c_int64_t), value     :: len_1d
  integer(c_int64_t) :: i
  real(c_double) :: acc

  acc = 0.0d0
  do i = 2, len_1d, 2
    acc = acc + a(i)
  end do
  out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
