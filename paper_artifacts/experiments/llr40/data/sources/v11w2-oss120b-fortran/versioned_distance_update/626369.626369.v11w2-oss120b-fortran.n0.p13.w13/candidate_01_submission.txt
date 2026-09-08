module versioned_distance_update_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine versioned_distance_update_fp64(a, b, c, K, LEN_1D) bind(C, name="versioned_distance_update_fp64")
    ! Implements the recurrence:
    !   a[i] = 0.75 * a[i-K] + b[i] * c[i]
    ! for i = K .. LEN_1D-1 (zero-based indexing). The first K elements of a are inputs and remain unchanged.
    ! The implementation parallelises across the K independent chains.
    implicit none
    real(c_double), intent(inout) :: a(*)          ! Output array, updated in place.
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(in)    :: c(*)
    integer(c_int64_t), value    :: LEN_1D
    integer(c_int64_t), value    :: K
    integer(c_int64_t) :: r, i

    if (K < 0_c_int64_t) then
    return
  else if (K == 0_c_int64_t) then
    !$omp parallel do default(none) shared(a, b, c, LEN_1D) private(i) schedule(static)
    do i = 1_c_int64_t, LEN_1D
      a(i) = 0.75_c_double * a(i) + b(i) * c(i)
    end do
    !$omp end parallel do
    return
  end if
    !$omp parallel do default(none) shared(a, b, c, LEN_1D, K) private(r, i) schedule(static)
    do r = 1_c_int64_t, K
      i = r + K
      do while (i <= LEN_1D)
        a(i) = 0.75_c_double * a(i - K) + b(i) * c(i)
        i = i + K
      end do
    end do
    !$omp end parallel do
  end subroutine versioned_distance_update_fp64
end module versioned_distance_update_mod
