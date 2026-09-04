! First-order linear recurrence with a VARIABLE coefficient:
!   y[i] = c[i] * y[i-1] + x[i]   (y[0] is the seed, already set).
! C ABI: void scan_affine_decay_fp64(double *y, double *c, double *x,
!                                    int64_t LEN_1D, uint8_t *ws, int64_t ws_bytes)
! NOTE: no libgfortran intrinsics (judge links the .so without libgfortran).

subroutine scan_affine_decay_fp64(y, c, x, len_1d, ws, ws_bytes) &
     bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  implicit none
  real(c_double), intent(inout) :: y(*)
  real(c_double), intent(in)    :: c(*)
  real(c_double), intent(in)    :: x(*)
  integer(c_int64_t), intent(in) :: len_1d
  type(c_ptr), intent(in)       :: ws
  integer(c_int64_t), intent(in) :: ws_bytes
  integer(c_int64_t) :: i

  do i = 2, len_1d
    y(i) = c(i) * y(i-1) + x(i)
  end do
end subroutine scan_affine_decay_fp64
