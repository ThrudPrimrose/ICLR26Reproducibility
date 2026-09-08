!> Fortran implementation of the TSVC kernel "s318" for double precision.
!>
!> The kernel computes the maximum absolute value of a 1-D input array
!> with a configurable stride (`inc`).  It also returns the zero-based
!> index (as a double) at which that maximum occurs.  The result is stored
!> in `result(1)` as `max_abs + index`.
!>
!> The reference C implementation (see /shared/tasks/tsvc_2_s318/tsvc_2_s318_reference.c)
!> is:
!>
!>   void tsvc_2_s318_fp64(const double *restrict a,
!>                         double *restrict result,
!>                         const int64_t LEN_1D,
!>                         const int64_t inc) {
!>       int64_t k, index;
!>       double maxv = 0.0, chksum = 0.0;
!>       k = 0; index = 0; maxv = fabs(a[0]);
!>       k += inc;
!>       for (int64_t i = 1; i < LEN_1D; ++i) {
!>           double v = fabs(a[k]);
!>           if (v > maxv) { index = i; maxv = v; }
!>           k += inc;
!>       }
!>       chksum = maxv + (double)(index);
!>       result[0] = chksum;
!>   }
!>
!> The Fortran version follows the same algorithm while applying OpenMP
!> parallel reduction to accelerate the max-reduction.  The index is then
!> obtained by a cheap sequential scan – the overhead is negligible compared
!> with the O(LEN_1D) work of the reduction.
!>
!> The subroutine is exposed with C bindings so the benchmark harness can
!> locate it by name.
!
module tsvc_2_s318_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s318_fp64(a, result, LEN_1D, inc) bind(C, name="tsvc_2_s318_fp64")
    ! Arguments matching the C prototype:
    !   void tsvc_2_s318_fp64(const double *restrict a,
    !                         double *restrict result,
    !                         const int64_t LEN_1D,
    !                         const int64_t inc)
    !
    ! a        – input array (const double*)
    ! result   – output array of length >= 1 (double*)
    ! LEN_1D   – number of logical elements
    ! inc      – stride between logical elements (in elements, not bytes)
    !
    ! All scalar arguments are passed by value from C, therefore the
    ! `value` attribute is required.
    real(C_DOUBLE), intent(in)  :: a(*)
    real(C_DOUBLE), intent(out) :: result(*)
    integer(C_INT64_T), value :: LEN_1D
    integer(C_INT64_T), value :: inc

    integer(C_INT64_T) :: i
    integer(C_INT64_T) :: idx
    real(C_DOUBLE) :: maxv, v

    ! Guard against zero length – the reference never calls with LEN_1D == 0,
    ! but we handle it defensively.
    if (LEN_1D <= 0) then
      result(1) = 0.0_C_DOUBLE
      return
    end if

    ! Initialise with the first element (C index 0 -> Fortran index 1).
    maxv = abs(a(1))
    idx  = 0_C_INT64_T

    !--------------------------------------------------------------------
    ! 1) Parallel reduction to find the maximum absolute value.
    !    The stride `inc` may be > 1, so we compute the address directly
    !    from the loop counter: a(1 + i*inc) corresponds to a[i*inc] in C.
    !--------------------------------------------------------------------
    !$omp parallel do private(i, v) reduction(max:maxv) schedule(static)
    do i = 1, LEN_1D - 1
      v = abs(a(1 + i * inc))
      if (v > maxv) maxv = v
    end do
    !$omp end parallel do

    !--------------------------------------------------------------------
    ! 2) Find the first index where the maximum occurs.
    !    This must be the *zero-based* C index, i.e. the value of `i`
    !    from the loop above when the condition is first true.
    !--------------------------------------------------------------------
    do i = 1, LEN_1D - 1
      if (abs(a(1 + i * inc)) == maxv) then
        idx = i
        exit
      end if
    end do

    ! Combine the two results as required by the reference.
    result(1) = maxv + real(idx, kind=C_DOUBLE)
  end subroutine tsvc_2_s318_fp64

end module tsvc_2_s318_mod

