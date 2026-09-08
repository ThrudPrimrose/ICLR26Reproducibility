subroutine tsvc_2_s323_fp64(a, b, cc, d, e, len_1d) bind(c, name='tsvc_2_s323_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), intent(inout) :: a(*), b(*)
  real(c_double), intent(in)    :: cc(*), d(*), e(*)
  integer(c_int64_t), intent(in), value :: len_1d

  integer(c_int64_t) :: i

  if (len_1d <= 1) return
  do i = 2, len_1d
     a(i) = b(i - 1) + cc(i) * d(i)
     b(i) = a(i) + cc(i) * e(i)
  end do
end subroutine tsvc_2_s323_fp64
