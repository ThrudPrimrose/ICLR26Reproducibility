subroutine tsvc_2_s115_fp64(a, aa, len_2d) bind(c, name='tsvc_2_s115_fp64')
  use iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a
  real(c_double), dimension(*), intent(in)    :: aa
  integer(c_int64_t), value, intent(in)       :: len_2d
  integer(c_int64_t) :: i, j

  do j = 0, len_2d - 1
     do i = j + 1, len_2d - 1
        a(i + 1) = a(i + 1) - aa(j * len_2d + i + 1) * a(j + 1)
     end do
  end do
end subroutine tsvc_2_s115_fp64
