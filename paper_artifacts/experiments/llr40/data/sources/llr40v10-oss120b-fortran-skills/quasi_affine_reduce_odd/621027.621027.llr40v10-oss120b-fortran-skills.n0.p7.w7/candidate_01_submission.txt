subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(C)
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(inout) :: out(1)
    real(c_double) :: acc
    integer(c_int64_t) :: i
    
    acc = 0.0_c_double
    !$omp parallel do simd reduction(+:acc)
    do i = 2, LEN_1D, 2
        acc = acc + a(i)
    end do
        out(1) = acc
end subroutine quasi_affine_reduce_odd_fp64
