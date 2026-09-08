subroutine tsvc_2_s323_fp64(a, b, c, d, e, len_1d) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in)    :: c(len_1d)
  real(c_double), intent(in)    :: d(len_1d)
  real(c_double), intent(in)    :: e(len_1d)

  integer(c_int64_t) :: i
  real(c_double) :: ai, t, u

  if (len_1d < 2) return
  ! Reference nest (bit-exact rounding: separate multiply and add, no FMA):
  !   t = c[i]*d[i];  a[i] = b[i-1] + t
  !   u = c[i]*e[i];  b[i] = a[i] + u
  ! The b(i-1) read is the previous iteration's write (a true carry), so the
  ! loop is a serial chain; temporaries force the reference's rounding.
  do i = 2, len_1d
    t = c(i) * d(i)
    ai = b(i - 1) + t
    a(i) = ai
    u = c(i) * e(i)
    b(i) = ai + u
  end do
end subroutine tsvc_2_s323_fp64
