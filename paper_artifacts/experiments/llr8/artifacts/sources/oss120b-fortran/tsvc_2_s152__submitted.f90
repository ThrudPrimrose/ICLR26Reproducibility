subroutine tsvc_2_s152_fp64(a, b, c, d, e, len_1d, workspace, workspace_bytes) bind(C, name="tsvc_2_s152_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: len_1d
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)
    real(c_double), intent(in) :: e(*)
    integer :: i
    real(c_double) :: tmp
    !$omp parallel do default(none) shared(a,b,c,d,e,len_1d) private(i,tmp)
    do i = 1, len_1d
        tmp = d(i) * e(i)
        b(i) = tmp
        a(i) = a(i) + tmp * c(i)
    end do
    !$omp end parallel do
end subroutine tsvc_2_s152_fp64
