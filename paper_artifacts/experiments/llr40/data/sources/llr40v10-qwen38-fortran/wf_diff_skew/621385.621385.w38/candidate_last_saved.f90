subroutine wf_diff_skew_fp64(a, len_2d, ws, ws_size) bind(c, name='wf_diff_skew_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), intent(in), value :: len_2d
  real(c_double), intent(inout) :: a(len_2d, len_2d)
  type(c_ptr), intent(in) :: ws
  integer(c_int64_t), intent(in), value :: ws_size
  integer(c_int64_t) :: i, j, n
  n = len_2d
  do i = 2, n
    do j = 1, n - 1
      a(j, i) = a(j, i) + a(j, i-1) + a(j+1, i-1)
    end do
  end do
end subroutine wf_diff_skew_fp64
