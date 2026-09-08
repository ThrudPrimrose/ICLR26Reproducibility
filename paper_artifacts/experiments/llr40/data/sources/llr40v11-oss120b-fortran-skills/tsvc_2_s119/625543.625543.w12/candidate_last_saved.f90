subroutine tsvc_2_s119_fp64(aa, bb, len_2d) bind(C, name="tsvc_2_s119_fp64")
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(inout) :: aa(len_2d, len_2d)
  real(c_double), intent(in) :: bb(len_2d, len_2d)
  integer(c_int64_t) :: d, i, j, i_start, j_start, n

  !$omp parallel do schedule(static,8) private(d,i,j,i_start,j_start,n)
  do d = -(len_2d - 2), len_2d - 2
    if (d >= 0) then
      i_start = d + 2
      j_start = 2
    else
      i_start = 2
      j_start = -d + 2
    end if
    n = len_2d - max(i_start, j_start) + 1
    do i = i_start, i_start + n - 1
      j = i - d
      aa(i, j) = aa(i-1, j-1) + bb(i, j)
    end do
  end do
  !$omp end parallel do

end subroutine tsvc_2_s119_fp64
