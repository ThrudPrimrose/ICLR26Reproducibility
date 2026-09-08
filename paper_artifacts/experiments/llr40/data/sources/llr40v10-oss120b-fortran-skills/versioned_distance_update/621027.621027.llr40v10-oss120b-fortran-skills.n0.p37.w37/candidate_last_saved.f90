subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D, workspace, workspace_size) bind(C)
    use iso_c_binding
    integer(c_int64_t), value, intent(in) :: K
    integer(c_int64_t), value, intent(in) :: LEN_1D
    integer(c_int64_t), value, intent(in) :: workspace_size
    type(c_ptr), value, intent(in) :: workspace
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    integer(c_int64_t) :: offset, i
    ! Parallelize over the K independent chains (offsets)
      !$omp parallel do private(offset,i) schedule(static)
    do offset = 1, K
        do i = offset + K, LEN_1D, K
            a(i) = 0.75d0 * a(i - K) + b(i) * c(i)
        end do
    end do
      !$omp end parallel do
end subroutine versioned_distance_update_fp64
