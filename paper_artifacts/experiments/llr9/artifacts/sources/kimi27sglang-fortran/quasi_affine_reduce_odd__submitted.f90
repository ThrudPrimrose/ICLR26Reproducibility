module quasi_affine_reduce_odd_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains

  subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(c, name='quasi_affine_reduce_odd_fp64')
    implicit none
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: acc
    integer(c_int64_t) :: i

    acc = 0.0_c_double
    do i = 2, LEN_1D, 2
      acc = acc + a(i)
    end do
    out(1) = acc
  end subroutine quasi_affine_reduce_odd_fp64

  subroutine quasi_affine_reduce_odd(a, out, LEN_1D) bind(c, name='quasi_affine_reduce_odd')
    implicit none
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: acc
    integer(c_int64_t) :: i

    acc = 0.0_c_double
    do i = 2, LEN_1D, 2
      acc = acc + a(i)
    end do
    out(1) = acc
  end subroutine quasi_affine_reduce_odd

end module quasi_affine_reduce_odd_mod
