module quasi_affine_reduce_odd_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(c, name='quasi_affine_reduce_odd_fp64')
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double) :: acc
    integer(c_int64_t) :: i, n
    n = LEN_1D
    acc = 0.0_c_double
    !$omp parallel do reduction(+:acc) schedule(static) num_threads(4)
    do i = 2_c_int64_t, n, 2_c_int64_t
      acc = acc + a(i)
    end do
    !$omp end parallel do
    out(1) = acc
  end subroutine
end module
