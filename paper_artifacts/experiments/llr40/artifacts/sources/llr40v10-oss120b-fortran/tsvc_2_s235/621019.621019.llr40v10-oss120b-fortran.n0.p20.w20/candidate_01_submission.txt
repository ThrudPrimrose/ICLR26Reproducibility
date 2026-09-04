! Fortran implementation of the TSVC tsvc_2 kernel "s235" (double precision).
! This mirrors the reference C implementation tsvc_2_s235_reference.c.
! The arrays a, b, c are length LEN_2D. The arrays aa and bb are flat LEN_2D*LEN_2D matrices
! stored in column-major order, matching the C indexing used in the reference.
!
module tsvc_2_s235_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains

  subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, LEN_2D) bind(C, name='tsvc_2_s235_fp64')
    ! Arguments correspond to the C prototype:
    !   double *restrict a,
    !   double *restrict aa,
    !   const double *restrict b,
    !   const double *restrict bb,
    !   const double *restrict c,
    !   const int64_t LEN_2D
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(inout) :: aa(*)
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(in)    :: bb(*)
    real(c_double), intent(in)    :: c(*)
    integer(c_int64_t), value    :: LEN_2D

    integer(c_int64_t) :: i, j
    integer(c_int64_t) :: idx

    !$omp parallel do default(none) shared(a,b,c,aa,bb,LEN_2D) private(i,j,idx)
    do i = 0, LEN_2D-1
      a(i+1) = a(i+1) + b(i+1) * c(i+1)
      do j = 1, LEN_2D-1
        idx = j * LEN_2D + i
        aa(idx+1) = aa(idx+1 - LEN_2D) + bb(idx+1) * a(i+1)
      end do
    end do
    !$omp end parallel do

  end subroutine tsvc_2_s235_fp64

end module tsvc_2_s235_mod
