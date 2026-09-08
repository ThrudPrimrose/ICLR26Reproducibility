subroutine tsvc_2_s233_fp64(aa, bb, cc, LEN_2D) bind(c, name='tsvc_2_s233_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: aa(LEN_2D, LEN_2D)
  real(c_double), intent(inout) :: bb(LEN_2D, LEN_2D)
  real(c_double), intent(in) :: cc(LEN_2D, LEN_2D)
  integer(c_int64_t) :: i, j

  do j = 9, LEN_2D
    do concurrent (i = 9:LEN_2D)
      aa(i, j) = aa(i, j - 1) + cc(i, j)
    end do
  end do

  do i = 9, LEN_2D
    do concurrent (j = 9:LEN_2D)
      bb(i, j) = bb(i - 1, j) + cc(i, j)
    end do
  end do
end subroutine tsvc_2_s233_fp64
