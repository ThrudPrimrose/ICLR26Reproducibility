! TSVC_2 s323:
!   for i = 1, LEN_1D-1:  a[i] = b[i-1] + c[i]*d[i];  b[i] = a[i] + c[i]*e[i]
!
! Exact-serial evaluation (0-based C semantics -> 1-based Fortran: i = 2..N).
! The expression forms mirror the C oracle so that gfortran's default FMA
! contraction reproduces the oracle's rounding exactly, element for element.
subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d, ws, ws_size) &
    bind(c, name='tsvc_2_s323_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: len_1d
  type(c_ptr), intent(in), value :: ws
  integer(c_int64_t), intent(in), value :: ws_size
  real(c_double), intent(inout) :: a(len_1d), b(len_1d)
  real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)

  integer(c_int64_t) :: i

  do i = 2, len_1d
    a(i) = b(i-1) + c(i)*d(i)
    b(i) = a(i) + c(i)*e(i)
  end do
end subroutine tsvc_2_s323_fp64
