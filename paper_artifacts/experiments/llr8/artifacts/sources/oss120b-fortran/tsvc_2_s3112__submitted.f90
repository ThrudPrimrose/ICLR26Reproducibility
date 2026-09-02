subroutine tsvc_2_s3112_fp64(a, b, LEN, workspace, workspace_bytes) bind(C, name='tsvc_2_s3112_fp64')
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN               ! Length of the arrays
    type(c_ptr), value :: workspace                ! Unused workspace pointer
    integer(c_int64_t), value :: workspace_bytes   ! Unused workspace size
    real(c_double), intent(in) :: a(*)             ! Input array
    real(c_double), intent(out) :: b(*)            ! Output prefix sums
    integer(c_int64_t) :: i
    real(c_double) :: sum

    sum = 0.0_c_double
    do i = 1, LEN
        sum = sum + a(i)
        b(i) = sum
    end do
end subroutine tsvc_2_s3112_fp64
