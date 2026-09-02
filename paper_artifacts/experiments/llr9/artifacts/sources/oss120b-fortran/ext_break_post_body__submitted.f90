! *********************************************************************
! ext_break_post_body kernel for HPCAgent-Bench (Fortran version)
! ---------------------------------------------------------------------
! The kernel receives three input arrays `a`, `b`, `c` each of length LEN_1D.
! It performs the computation:
!   a[i] = a[i] + b[i] * c[i]
! for i = 0 .. N-1 (0‑based), and after each update checks the guard
!   if c[i] > b[i] then break.
! In Fortran (1‑based indexing) this corresponds to updating `a(i)` for all
!   i = 1 .. break_idx (inclusive), where `break_idx` is the smallest index
!   for which c(i) > b(i). If no such index exists the loop runs over the
!   entire range.
!
! The reference implementation (Python/NumPy) is a serial loop with an
! early break. To obtain speed‑up on multi‑core CPUs we employ a two‑pass
! algorithm:
!   1. In parallel find the smallest index where the guard holds (a MIN
!      reduction over candidate indices).
!   2. In parallel update `a` for all indices up to that point.
!
! The subroutine follows the name and argument order expected by the
! benchmark harness:
!   subroutine ext_break_post_body(a, b, c, LEN_1D)
! where
!     a, b, c : real(8) arrays, size LEN_1D,
!               intent(inout) for a, intent(in) for b and c,
!     LEN_1D : integer(8) scalar, intent(in).
!
! OpenMP is used for parallelism; the code compiles with the benchmark
! flags (-O3 -march=native -fopenmp ...). The implementation is thread‑safe
! and conforms to the required semantics.
! *********************************************************************

subroutine ext_break_post_body_fp64(a, b, c, LEN_1D, workspace, workspace_bytes) bind(C, name="ext_break_post_body_fp64")
    use iso_c_binding, only: c_int64_t, c_double, c_ptr
    use omp_lib
    implicit none

    ! Arguments
    integer(c_int64_t), value :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D)
    real(c_double), intent(in) :: c(LEN_1D)
    type(c_ptr), value :: workspace          ! unused
    integer(c_int64_t), value :: workspace_bytes ! unused

    ! Local variables
    integer(c_int64_t) :: i
    integer(c_int64_t) :: break_idx
    integer(c_int64_t), parameter :: sentinel = huge(0_c_int64_t)

    ! Edge case: empty or negative length – nothing to do
    if (LEN_1D <= 0_c_int64_t) then
        return
    end if

    ! -----------------------------------------------------------------
    ! Phase 1: Find the smallest index i (1‑based) where c(i) > b(i).
    ! Use a parallel reduction with the MIN operator. The reduction
    ! identity for MIN is the largest representable integer, so initializing
    ! break_idx to sentinel is sufficient.
    ! -----------------------------------------------------------------
    break_idx = sentinel
    !$omp parallel default(none) private(i) reduction(min:break_idx) shared(c, b, LEN_1D)
    !$omp do schedule(static)
    do i = 1, LEN_1D
        if (c(i) > b(i)) then
            break_idx = i
        end if
    end do
    !$omp end do
    !$omp end parallel

    ! If no element satisfies the guard, treat it as a break after the last
    ! element (i.e., the whole range is processed).
    if (break_idx > LEN_1D) then
        break_idx = LEN_1D + 1_c_int64_t
    end if

    ! -----------------------------------------------------------------
    ! Phase 2: Update a(i) = a(i) + b(i) * c(i) for all i <= break_idx.
    ! The upper bound is min(break_idx, LEN_1D).
    ! -----------------------------------------------------------------
    if (break_idx > 0_c_int64_t) then
        !$omp parallel do default(none) private(i) shared(a, b, c, break_idx, LEN_1D)
        do i = 1, min(break_idx, LEN_1D)
            a(i) = a(i) + b(i) * c(i)
        end do
        !$omp end parallel do
    end if

    ! No explicit return value – the updated array `a` is the result.
end subroutine ext_break_post_body_fp64

