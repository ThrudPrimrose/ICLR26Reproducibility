subroutine tsvc_2_s2233_fp64(aa, bb, cc, LEN_2D) bind(c)
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), intent(in), value :: LEN_2D
  real(c_double), intent(inout) :: aa(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(inout) :: bb(0:LEN_2D-1, 0:LEN_2D-1)
  real(c_double), intent(in) :: cc(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: i, j

  do i = 8, LEN_2D - 1
    do j = 8, LEN_2D - 1
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
    do j = 8, LEN_2D - 1
      bb(j, i) = bb(j, i - 1) + cc(j, i)
    end do
  end do
end subroutine tsvc_2_s2233_fp64
