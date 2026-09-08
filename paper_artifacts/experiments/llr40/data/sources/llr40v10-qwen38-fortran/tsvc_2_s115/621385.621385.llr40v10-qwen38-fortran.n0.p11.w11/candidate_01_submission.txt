subroutine tsvc_2_s115_fp64(a, aa, len_2d) bind(c, name="tsvc_2_s115_fp64")
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
  integer(c_int64_t), value :: len_2d
  real(c_double), intent(inout), dimension(*) :: a
  real(c_double), intent(in), dimension(*) :: aa

  integer :: n, j, i
  n = int(len_2d)
  do j = 0, n - 1
     do i = j + 1, n - 1
        a(i + 1) = a(i + 1) - aa(j * n + i + 1) * a(j + 1)
     end do
  end do
end subroutine tsvc_2_s115_fp64
