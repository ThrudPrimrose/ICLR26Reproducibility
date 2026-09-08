! Fortran implementation of the TSVC tsvc_2 s2233 kernel (double precision)
! Implements the same algorithm as the C reference and NumPy reference.
! The subroutine follows the C ABI and is exported with the exact name
! tsvc_2_s2233_fp64 for the judge to locate.

subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(C)
  use iso_c_binding, only: c_double, c_int64_t
  implicit none

  integer(c_int64_t), value, intent(in) :: LEN_2D
  ! Declare the arrays with a 0-based lower bound to match the C indexing.
  real(c_double), intent(inout) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(inout) :: bb(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(in)    :: cc(0:LEN_2D-1, 0:LEN_2D-1)

  integer(c_int64_t) :: i, j

  ! First recurrence: column-wise update of aa.
  ! The outer loop (i) can be parallelised because each column is
  ! independent of the others.
  !$omp parallel do default(none) schedule(static) shared(aa, cc, LEN_2D) private(i, j)
  do i = 8, LEN_2D-1
    do j = 8, LEN_2D-1
      aa(i, j) = aa(i, j-1) + cc(i, j)
    end do
  end do
  !$omp end parallel do

  ! Second recurrence: row-wise update of bb.
  ! The outer loop i has a dependence on the previous row, so it stays
  ! serial. The inner loop over j is vectorisable.
  do i = 8, LEN_2D-1
    !$omp simd
    do j = 8, LEN_2D-1
      bb(j, i) = bb(j, i-1) + cc(j, i)
    end do
    !$omp end simd
  end do

end subroutine tsvc_2_s2233_fp64
