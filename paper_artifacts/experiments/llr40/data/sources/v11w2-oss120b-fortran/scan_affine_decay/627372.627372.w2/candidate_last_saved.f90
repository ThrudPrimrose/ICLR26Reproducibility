module scan_affine_decay_mod
  use iso_c_binding
  implicit none
contains
  subroutine scan_affine_decay_fp64(c, x, y, LEN_1D) bind(C, name="scan_affine_decay_fp64")
    ! Arguments: c (coefficients), x (input), y (output)
    real(c_double), intent(in) :: c(*)
    real(c_double), intent(in) :: x(*)
    real(c_double), intent(inout) :: y(*)
    integer(c_int64_t), value :: LEN_1D
    integer(c_int64_t) :: i
    if (LEN_1D <= 0) return
    ! Seed: y[0] = x[0]
    y(1) = x(1)
    if (LEN_1D == 1) return
    do i = 2, LEN_1D
        y(i) = c(i) * y(i-1) + x(i)
    end do
  end subroutine scan_affine_decay_fp64
end module scan_affine_decay_mod
