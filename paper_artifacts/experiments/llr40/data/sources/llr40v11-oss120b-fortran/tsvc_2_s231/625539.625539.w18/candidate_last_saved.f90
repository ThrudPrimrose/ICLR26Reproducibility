!> @file tsvc_2_s231.f90
!> Hand-written Fortran implementation of the TSVC 2 kernel "s231" (double precision).
!>
!> The reference C implementation (see /shared/tasks/tsvc_2_s231/tsvc_2_s231_reference.c)
!> performs a column-wise prefix sum:
!>   aa[j,i] = aa[j-1,i] + bb[j,i]  for j = 1 ... LEN_2D-1, i = 0 ... LEN_2D-1
!> where the arrays are stored in column-major order (the same layout used by Fortran).
!>
!> This subroutine follows the C ABI expected by the benchmark:
!>   void tsvc_2_s231_fp64(double *restrict aa,
!>                         const double *restrict bb,
!>                         const int64_t LEN_2D);
!>
!> The Fortran interface uses BIND(C) with the exact name and a VALUE argument for
!> LEN_2D so that the calling convention matches the C signature.
!>
!> The outer loop over columns (index i) is parallelised with OpenMP.  The inner
!> loop carries a true recurrence and therefore cannot be parallelised, but the
!> column-wise decomposition gives good scaling on typical multi-core CPUs.
!>
!> The implementation is deliberately straightforward – readability aids correctness
!> and the auto-vectorisation performed by the compiler on the inner loop yields the
!> best achievable performance for this algorithmic pattern.
!
module tsvc_2_s231_mod
  use iso_c_binding, only: c_double, c_int64_t
  use omp_lib
  implicit none
contains

  subroutine tsvc_2_s231_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s231_fp64")
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: bb(*)
    integer(c_int64_t), value    :: LEN_2D
    integer(c_int64_t) :: i, j, N, base, idx
    real(c_double) :: acc
    N = LEN_2D
    !$omp parallel do default(none) shared(aa, bb, N) private(i, j, base, idx, acc)
    do i = 0, N-1
      base = i * N
      acc = aa(base + 1)
      do j = 1, N-1
        idx = base + j + 1
        acc = acc + bb(idx)
        aa(idx) = acc
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s231_fp64

end module tsvc_2_s231_mod

