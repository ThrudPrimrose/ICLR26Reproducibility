subroutine wf_diff_skew_fp64(a, len_2d) bind(c, name="wf_diff_skew_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  integer(c_int64_t) :: i, j
  do i = 2, len_2d
     do j = 1, len_2d - 1
        a(j, i) = a(j, i) + a(j, i-1) + a(j+1, i-1)
     end do
  end do
end subroutine
