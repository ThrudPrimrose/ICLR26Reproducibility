subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_1D
  real(c_double), intent(in) :: a(LEN_1D)
  real(c_double), intent(out) :: out(1)
  integer(c_int64_t) :: i
  real(c_double) :: acc

  acc = 0.0d0
  !$omp parallel do reduction(+:acc) default(none) shared(a, LEN_1D) private(i)
  do i = 2, LEN_1D, 2
    acc = acc + a(i)
  end do
  out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
