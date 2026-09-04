module quasi_affine_reduce_odd_mod
  use iso_c_binding
  implicit none
contains
  subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(C, name="quasi_affine_reduce_odd_fp64")
    implicit none
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t) :: i
    real(c_double) :: acc
    acc = 0.0_c_double
    !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(+:acc) schedule(static)
    do i = 2, LEN_1D, 2
       acc = acc + a(i)
    end do
    !$omp end parallel do
    out(1) = acc
  end subroutine quasi_affine_reduce_odd_fp64
end module quasi_affine_reduce_odd_mod
