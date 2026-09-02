subroutine tsvc_2_s323(a, b, c, d, e, len_1d, workspace, workspace_size) bind(C, name='tsvc_2_s323_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d), d(len_1d), e(len_1d)
  integer(c_int64_t), value, intent(in) :: workspace_size
  integer(c_int8_t), intent(inout) :: workspace(workspace_size)
  integer(c_int64_t) :: i, n

  n = len_1d
  ! Coupled recurrence: a(i)=b(i-1)+c(i)*d(i); b(i)=a(i)+c(i)*e(i).
  ! b is a serial scan of c(i)*(d(i)+e(i)) seeded by b(1); the coupling forces a
  ! bit-exact serial chain (tight fp64 grading), so replicate the reference order.
  do i = 2, n
    a(i) = b(i-1) + c(i)*d(i)
    b(i) = a(i) + c(i)*e(i)
  end do
end subroutine tsvc_2_s323
