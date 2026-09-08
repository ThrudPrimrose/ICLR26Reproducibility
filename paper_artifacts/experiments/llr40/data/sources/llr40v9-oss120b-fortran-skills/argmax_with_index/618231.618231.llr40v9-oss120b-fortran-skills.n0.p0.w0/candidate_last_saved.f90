! Fortran implementation of the argmax_with_index_fp64 kernel.
! Finds the maximum value in a 1D double-precision array and its index.
! The output index follows the zero‑based convention expected by the reference.
!
subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none

    ! Arguments – scalars must be VALUE and appear first.
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in)           :: a(LEN_1D)
    integer(c_int64_t), intent(out)     :: out_index(1)
    real(c_double), intent(out)          :: out_value(1)

    integer(c_int64_t) :: i
    real(c_double)    :: maxv
    integer(c_int64_t) :: idx

    ! -----------------------------------------------------------------
    ! Phase 1 – parallel reduction to obtain the maximum value.
    ! -----------------------------------------------------------------
    maxv = -huge(0.0_c_double)
    !$omp parallel do default(none) shared(a, LEN_1D) private(i) reduction(max:maxv) schedule(static)
    do i = 1, LEN_1D
        if (a(i) > maxv) maxv = a(i)
    end do
    !$omp end parallel do

    ! -----------------------------------------------------------------
    ! Phase 2 – serial scan to locate the first occurrence of the maximum.
    ! -----------------------------------------------------------------
    idx = 0
    do i = 1, LEN_1D
        if (a(i) == maxv) then
            idx = i
            exit
        end if
    end do

    out_value(1) = maxv
    out_index(1) = idx
end subroutine argmax_with_index_fp64
