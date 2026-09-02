subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C)
    use iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(1)
    type(c_ptr), intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    integer(c_int64_t) :: i
    real(c_double) :: sum
    sum = 0.0d0
    !$omp parallel do reduction(+:sum) schedule(static)
    do i = 2, LEN_1D, 2
        sum = sum + a(i)
    end do
    !$omp end parallel do
    out(1) = sum
end subroutine quasi_affine_reduce_odd_fp64
