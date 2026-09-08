subroutine quasi_affine_reduce_odd_fp64(a, out, LEN_1D, workspace, workspace_size) bind(c, name='quasi_affine_reduce_odd_fp64')
    use iso_c_binding, only: c_double, c_int64_t, c_ptr
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(LEN_1D)
    real(c_double), intent(out) :: out(1)
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size

    out(1) = sum(a(2:LEN_1D:2))
end subroutine quasi_affine_reduce_odd_fp64
