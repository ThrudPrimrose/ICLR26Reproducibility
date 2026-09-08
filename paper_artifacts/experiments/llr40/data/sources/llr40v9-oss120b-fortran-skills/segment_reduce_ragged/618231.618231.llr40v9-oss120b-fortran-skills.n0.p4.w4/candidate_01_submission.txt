! segment_reduce_ragged kernel implementation in Fortran
!
! Computes a segmented dot product (reduction) over ragged CSR-like data.
! For each segment s (0 <= s < NSEG) the result is
!   out(s) = sum_{e=row_ptr[s]}^{row_ptr[s+1]-1} val(e) * w(e)
!
! The input arrays follow NumPy conventions: row_ptr contains zero‑based offsets,
! val and w are double‑precision values, and out stores the per‑segment results.
! Fortran arrays are 1‑based, therefore we add 1 to the offsets when indexing.
!
! The required C‑ABI entry point is segment_reduce_ragged_fp64. The harness expects
! the arguments in the following order (matching its C prototype):
!   double *out,
!   int64_t *row_ptr,
!   double *val,
!   double *w,
!   int64_t NSEG,
!   uint8_t *workspace,
!   int64_t workspace_size
!
! The workspace arguments are unused by this kernel but must be present.
!
! Parallelism: the outer loop over segments is parallelised with a dynamic schedule
! to balance the irregular work across threads.  The inner loop is vectorised via
! an OpenMP SIMD reduction.
!
subroutine segment_reduce_ragged_fp64(out, row_ptr, val, w, NSEG, workspace, workspace_size) bind(C)
    use iso_c_binding
    use omp_lib
    implicit none

    !--- Argument declarations -------------------------------------------------
    integer(c_int64_t), value, intent(in) :: NSEG
    type(c_ptr), value, intent(in)        :: workspace
    integer(c_int64_t), value, intent(in) :: workspace_size

    real(c_double), intent(out)   :: out(NSEG)                ! output per segment
    integer(c_int64_t), intent(in) :: row_ptr(NSEG+1)         ! zero‑based offsets
    real(c_double), intent(in)    :: val(*)                  ! values (length = row_ptr(NSEG+1))
    real(c_double), intent(in)    :: w(*)                    ! weights (same length as val)

    !--- Local variables ------------------------------------------------------
    integer(c_int64_t) :: s, e
    integer(c_int64_t) :: start_idx, end_idx
    real(c_double)    :: acc

    ! Parallel over segments – dynamic schedule handles ragged lengths.
    !$omp parallel do schedule(guided) private(acc, start_idx, end_idx)
    do s = 1, NSEG
        acc = 0.0_c_double
        ! Convert zero‑based offsets to Fortran 1‑based indices.
        start_idx = row_ptr(s) + 1_c_int64_t
        end_idx   = row_ptr(s+1)
        ! The inner loop may be empty for zero‑length segments.
        !$omp simd reduction(+:acc)
        do e = start_idx, end_idx
            acc = acc + val(e) * w(e)
        end do
        out(s) = acc
    end do
    !$omp end parallel do

end subroutine segment_reduce_ragged_fp64
