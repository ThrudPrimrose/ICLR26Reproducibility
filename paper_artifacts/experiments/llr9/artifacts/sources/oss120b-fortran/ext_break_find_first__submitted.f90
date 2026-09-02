! *********************************************************************
! ext_break_find_first kernel for HPCAgent-Bench (Fortran version)
! ---------------------------------------------------------------------
! The kernel receives four input arrays `a`, `b`, `c`, `d` each of length LEN_1D.
! It scans `d` for the first negative element, breaks the loop when such
! an element is found, and for all preceding elements updates `a` with
! a[i] = a[i] + b[i] * c[i].
!
! The reference implementation (Python/NumPy) performs a serial scan with an
! early break. For large arrays, a parallel two‑pass algorithm can achieve a
! speed‑up while preserving the same semantics.
!
! The subroutine follows the name and argument order expected by the
! benchmark harness:
!   subroutine ext_break_find_first(a, b, c, d, LEN_1D)
! where
!   a, b, c, d : real(8) arrays, size LEN_1D,
!                intent(inout) for a, intent(in) for the others,
!   LEN_1D    : integer(8) scalar, intent(in).
!
! The implementation uses OpenMP to find the first negative index in `d`
! via a parallel reduction and then updates the prefix of `a` in a parallel
! loop. This approach scales with the number of cores while preserving the
! benchmark's required semantics.
! *********************************************************************

subroutine ext_break_find_first_fp64(a, b, c, d, LEN_1D, workspace, workspace_bytes) bind(C, name="ext_break_find_first_fp64")
    use iso_c_binding, only: c_int64_t, c_double, c_ptr
    use omp_lib
    implicit none

    ! Arguments
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: d(*)

    ! Local variables
    integer(c_int64_t) :: i
    integer(c_int64_t) :: break_idx
    integer(c_int64_t), parameter :: sentinel = huge(0_c_int64_t)

    ! Edge case: empty array – nothing to do
    if (LEN_1D <= 0_c_int64_t) then
        return
    end if

    ! -----------------------------------------------------------------
    ! Phase 1: Find the smallest index i where d(i) < 0 using a parallel
    ! reduction with the MIN operator. The reduction's private copies are
    ! initialised to the maximum representable integer, so after the region
    ! `break_idx` holds the minimum matching index or a sentinel value if no
    ! element satisfies the condition.
    ! -----------------------------------------------------------------
    break_idx = sentinel
    !$omp parallel default(none) private(i) reduction(min:break_idx) shared(d, LEN_1D)
    !$omp do schedule(static)
    do i = 1, LEN_1D
        if (d(i) < 0.0_c_double) then
            break_idx = i
        end if
    end do
    !$omp end do
    !$omp end parallel

    ! If no negative was found (should not happen for the benchmark's
    ! generated inputs except for LEN_1D <= 1), treat it as a break after the
    ! last element.
    if (break_idx > LEN_1D) then
        break_idx = LEN_1D + 1_c_int64_t
    end if

    ! -----------------------------------------------------------------
    ! Phase 2: Update a(i) for all indices before the break point.
    ! -----------------------------------------------------------------
    if (break_idx > 1_c_int64_t) then
        !$omp parallel do default(none) private(i) shared(a, b, c, break_idx)
        do i = 1, break_idx - 1
            a(i) = a(i) + b(i) * c(i)
        end do
        !$omp end parallel do
    end if

    ! No explicit return value – results are stored in the updated array `a`.
end subroutine ext_break_find_first_fp64
