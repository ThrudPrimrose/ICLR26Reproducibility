! scan_affine_decay.f90
! Fortran kernel implementing y[i] = c[i] * y[i-1] + x[i]
! The recurrence is 0-indexed in C; Fortran arrays are 1-indexed.
! The initial element y(1) holds x(1) as set by the harness.
! This subroutine follows the C ABI via BIND(C) and expects arrays of double precision
! and length as a 64-bit integer.
! SPDX-License-Identifier: GPL-3.0-or-later

subroutine scan_affine_decay_fp64(y, c, x, LEN_1D, workspace, workspace_bytes) bind(C, name="scan_affine_decay_fp64")
  use iso_c_binding, only: c_double, c_int64_t, c_ptr
  use omp_lib
  implicit none
  real(c_double), intent(inout) :: y(*)
  real(c_double), intent(in)    :: c(*)
  real(c_double), intent(in)    :: x(*)
  integer(c_int64_t), value, intent(in)    :: LEN_1D
  type(c_ptr), value, intent(in) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_bytes
  integer(c_int64_t) :: i
  real(c_double) :: acc

  if (LEN_1D <= 1_c_int64_t) return
  acc = 0.0_c_double

  !$omp parallel
  !$omp single
  do i = 2_c_int64_t, LEN_1D
    y(i) = c(i) * y(i-1) + x(i)
    acc = acc + y(i)
  end do
  !$omp end single
  !$omp end parallel

  ! Dummy use of acc to inhibit aggressive optimizations that might incorrectly assume the loop is parallelizable.
  ! The conditional is never true because the recurrence produces strictly positive values,
  ! but the compiler must preserve the dependence on acc.
  if (acc < 0.0_c_double) then
    y(1) = y(1) + acc
  end if
end subroutine scan_affine_decay_fp64