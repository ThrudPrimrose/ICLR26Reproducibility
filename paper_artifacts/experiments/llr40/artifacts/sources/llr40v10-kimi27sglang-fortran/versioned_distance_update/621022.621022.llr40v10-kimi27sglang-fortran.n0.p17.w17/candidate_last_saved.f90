subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_size) &
    bind(c, name="versioned_distance_update_fp64")
    use iso_c_binding, only: c_double, c_int64_t
    implicit none
    real(c_double), dimension(*), intent(inout) :: a, b, c
    integer(c_int64_t), value :: K, LEN_1D
    real(c_double), dimension(*), intent(inout) :: workspace
    integer(c_int64_t), value :: workspace_size
    integer(c_int64_t) :: i
    real(c_double) :: prev

    if (K == 1_c_int64_t) then
        prev = a(1)
        do i = 2, LEN_1D
            prev = 0.75_c_double * prev + b(i) * c(i)
            a(i) = prev
        end do
    else
        do i = K + 1, LEN_1D
            a(i) = 0.75_c_double * a(i - K) + b(i) * c(i)
        end do
    end if
end subroutine versioned_distance_update_fp64
