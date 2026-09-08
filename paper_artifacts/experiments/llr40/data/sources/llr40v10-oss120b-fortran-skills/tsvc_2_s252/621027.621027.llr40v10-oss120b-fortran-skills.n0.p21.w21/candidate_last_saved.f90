subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D, workspace, workspace_size) bind(C, name="tsvc_2_s252_fp64")
    use iso_c_binding
    use omp_lib
    implicit none
    ! Arguments
    integer(c_int64_t), value, intent(in) :: LEN_1D
    type(c_ptr), value, intent(in) :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    ! Local variables
    real(c_double), allocatable :: s(:)
    integer(c_int64_t) :: i
    ! Cast workspace to a temporary array for products
    allocate(s(LEN_1D))
    ! Compute products b*c into s
    !$omp parallel do schedule(static)
    do i = 1, LEN_1D
        s(i) = b(i) * c(i)
    end do
    !$omp end parallel do
    ! Compute a: first element has no previous product
    if (LEN_1D >= 1) then
        a(1) = s(1)
    end if
    if (LEN_1D > 1) then
        !$omp parallel do schedule(static)
        do i = 2, LEN_1D
            a(i) = s(i) + s(i-1)
        end do
        !$omp end parallel do
    end if
    ! Deallocate pointer (optional, not needed as it points to external memory)
    deallocate(s)
end subroutine tsvc_2_s252_fp64
