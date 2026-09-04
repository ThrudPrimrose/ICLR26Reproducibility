subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D) bind(C, name="versioned_distance_update_fp64")
    use iso_c_binding
    implicit none
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: c(*)
    integer(c_int64_t), value :: K
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: offset, i

    ! If the distance K is greater than or equal to the array length, no updates are needed.
    if (K >= LEN_1D) return

    !$omp parallel do default(none) shared(a,b,c,K,LEN_1D) private(offset,i)
    do offset = 1, K
        i = offset + K
        do while (i <= LEN_1D)
            a(i) = 0.75d0 * a(i-K) + b(i) * c(i)
            i = i + K
        end do
    end do
    !$omp end parallel do
end subroutine versioned_distance_update_fp64
