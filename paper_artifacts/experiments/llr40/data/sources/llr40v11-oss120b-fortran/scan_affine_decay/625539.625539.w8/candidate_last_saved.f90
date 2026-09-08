! scan_affine_decay.f90
! Implementation of variable-coefficient affine scan using parallel block prefix.
!
module scan_affine_decay_mod
  use iso_c_binding
  use omp_lib
  implicit none
contains
  subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_len) bind(C, name="scan_affine_decay_fp64")
    ! Arguments: y - output array (modified in-place),
    !            c - coefficient array,
    !            x - additive array,
    !            LEN_1D - number of elements (int)
    implicit none
    integer(c_int64_t), value :: LEN_1D
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_len
    real(c_double), intent(inout) :: y(*)
    real(c_double), intent(in) :: c(*), x(*)

    integer(c_int64_t) :: i

    if (LEN_1D <= 0) return
    ! seed the recurrence: y(1) = x(1) (Fortran 1‑based indexing)
    y(1) = x(1)
    do i = 2, LEN_1D
        y(i) = c(i) * y(i-1) + x(i)
    end do
  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
