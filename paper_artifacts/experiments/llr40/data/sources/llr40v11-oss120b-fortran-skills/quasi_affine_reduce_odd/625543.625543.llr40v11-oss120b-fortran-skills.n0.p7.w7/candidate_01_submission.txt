subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D) bind(C, name="quasi_affine_reduce_odd_fp64")
    use iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: out(1)
    real(c_double) :: acc
    integer(c_int64_t) :: i

    acc = 0.0_c_double

    !$omp parallel do reduction(+:acc) schedule(static)
    do i = 2, LEN_1D, 2
        acc = acc + a(i)
    end do
    ! No explicit barrier needed; reduction handles it
    out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
