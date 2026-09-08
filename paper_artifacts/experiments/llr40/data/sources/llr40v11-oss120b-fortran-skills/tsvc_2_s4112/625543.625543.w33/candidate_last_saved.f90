subroutine tsvc_2_s4112_fp64(a, b, ip, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s4112_fp64")
    use iso_c_binding
    use omp_lib
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    integer(c_int32_t), intent(in) :: ip(LEN_1D)
    integer(c_int64_t) :: i
    !$omp parallel do schedule(static) default(none) shared(a,b,ip,LEN_1D) private(i)
    do i = 1, LEN_1D
        a(i) = a(i) + b(ip(i) + 1) * 2.0_c_double
    end do
    !$omp end parallel do
end subroutine tsvc_2_s4112_fp64
