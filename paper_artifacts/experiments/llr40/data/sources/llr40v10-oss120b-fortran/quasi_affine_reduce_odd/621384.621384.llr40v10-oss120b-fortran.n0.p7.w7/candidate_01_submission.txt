module quasi_affine_reduce_odd_mod
  use iso_c_binding
  implicit none
contains
  subroutine quasi_affine_reduce_odd_fp64(a, out, len_1d) bind(C, name="quasi_affine_reduce_odd_fp64")
    ! Arguments: a - input array of double, out - output array of double (size 1), len_1d - length of a
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t), value :: len_1d
    real(c_double) :: acc
    integer(c_int64_t) :: i

    acc = 0.0_c_double
    !$omp parallel do reduction(+:acc) schedule(static)
    do i = 2_c_int64_t, len_1d, 2_c_int64_t
      acc = acc + a(i)
    end do
    !$omp end parallel do
    out(1) = acc
  end subroutine quasi_affine_reduce_odd_fp64
end module quasi_affine_reduce_odd_mod
