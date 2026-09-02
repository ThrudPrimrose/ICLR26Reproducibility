subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d, workspace, workspace_size) &
     bind(C, name="tsvc_2_s2710_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(inout) :: b(len_1d)
  real(c_double), intent(inout) :: c(len_1d)
  real(c_double), intent(in)    :: d(len_1d)
  real(c_double), intent(in)    :: e(len_1d)
  real(c_double), intent(in)    :: x(len_1d)
  type(c_ptr), intent(in)       :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size

  logical :: big, xp, t
  real(c_double) :: ai, bi, ci, di, ei, c_if, c_else
  integer(c_int64_t) :: i

  big = len_1d > 10
  xp  = x(1) > 0.0d0

!$omp parallel do schedule(static)
  do i = 1, len_1d
    ai = a(i); bi = b(i); ci = c(i); di = d(i); ei = e(i)
    t = ai > bi
    a(i) = merge(ai + bi*di, ai, t)
    b(i) = merge(bi, ai + ei*ei, t)
    c_if   = merge(ci + di*di, di*ei + 1.0d0, big)
    c_else = merge(ai + di*di, ci + ei*ei, xp)
    c(i)   = merge(c_if, c_else, t)
  end do
!$omp end parallel do
end subroutine
