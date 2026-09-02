! wf_diff_skew.f90
! Fortran implementation of the wf_diff_skew kernel for the OptArena benchmark.
! The driver expects a C‑linkage entry point with the suffix "_fp64" and the
! signature:
!   void wf_diff_skew_fp64(double *a, int64_t len_2d, uint8_t *workspace,
!                         int64_t workspace_bytes);
! The workspace arguments are unused for this kernel but are required by the
! harness.  The array "a" is a square double‑precision matrix stored in
! column‑major order (Fortran layout) with dimensions (len_2d, len_2d).
!
! The kernel computes:
!   a(i,j) = a(i,j) + a(i-1,j) + a(i-1,j+1)
! for i = 2 .. len_2d and j = 1 .. len_2d‑1 (1‑based indexing).  This matches
! the reference implementation in /shared/tasks/wf_diff_skew/wf_diff_skew_numpy.py.
!
subroutine wf_diff_skew_fp64(a, len_2d, workspace, workspace_bytes) bind(C, name="wf_diff_skew_fp64")
    use iso_c_binding
    implicit none

    ! Arguments coming from the harness
    type(c_ptr), value :: a               ! pointer to the matrix data (double*)
    integer(c_int64_t), value :: len_2d   ! matrix dimension
    type(c_ptr), value :: workspace       ! not used, required by the API
    integer(c_int64_t), value :: workspace_bytes

    ! Fortran view of the data
    real(c_double), pointer :: arr(:)
    integer(c_int) :: n, i, j, idx, size

    ! Convert the 64‑bit length to the 32‑bit integer used for loop counters.
    n = int(len_2d, kind=c_int)

    ! Map the raw pointer to a 1‑D Fortran array.
    size = n * n
    call c_f_pointer(a, arr, [size])

    do i = 2, n
        do j = 1, n-1
            idx = (i-1) * n + j
            arr(idx) = arr(idx) + arr(idx - n) + arr(idx - n + 1)
        end do
    end do

end subroutine wf_diff_skew_fp64
