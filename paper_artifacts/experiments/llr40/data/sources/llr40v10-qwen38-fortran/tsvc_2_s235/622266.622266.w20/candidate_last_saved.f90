subroutine tsvc_2_s235_fp64(a, aa, b, bb, c, len_2d) bind(c, name="tsvc_2_s235_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d)
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: b(len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  real(c_double), intent(in) :: c(len_2d)
  integer :: n, i, j
  n = int(len_2d, 4)
  write(*, "(a,i0)") " PROBE_N=", n
  flush(6)
  do i = 1, n
     a(i) = a(i) + b(i) * c(i)
  end do
  do j = 1, n - 1
     do i = 1, n
        aa(i, j + 1) = aa(i, j) + bb(i, j + 1) * a(i)
     end do
  end do
end subroutine
